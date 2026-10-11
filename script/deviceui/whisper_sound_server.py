"""Loopback Whisper proxy with serialized, request-bound shared-encoder tags.

The native child is private to this adapter. Its feature file is removed before
each request and consumed inside the same lock, so no caller can observe another
request's features. The ASR response survives missing/invalid tag observations.
No classifier threshold decides whether words or a voice should be admitted.
"""
import argparse
from email.parser import BytesParser
from email.policy import default
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.request
import wave

from sound_event import LABELS, validate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audio_seconds(body, content_type):
    message = BytesParser(policy=default).parsebytes(
        ('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + body)
    files = [part for part in message.iter_parts()
             if part.get_param('name', header='content-disposition') == 'file']
    if len(files) != 1:
        raise ValueError('Exactly one PCM WAV is required')
    with wave.open(io.BytesIO(files[0].get_payload(decode=True))) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('Sound tagging requires 16 kHz mono PCM16 WAV')
        seconds = wav.getnframes()/16000
    if not 0 < seconds <= 30:
        raise ValueError('Sound tagging requires an actual span of up to 30 seconds')
    formats = [part.get_payload(decode=True).decode() for part in message.iter_parts()
               if part.get_param('name', header='content-disposition') == 'response_format']
    if formats != ['vtt']:
        raise ValueError('Tagged requests require VTT timestamps')
    return seconds


class AcousticHead:
    def __init__(self, settings):
        import numpy as np
        import onnxruntime as ort
        if digest(settings['head']) != settings['head_sha256']:
            raise ValueError('Acoustic head hash mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(settings['head'], sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.shape = tuple(settings['feature_shape'])
        if self.session.get_inputs()[0].shape != list(self.shape):
            raise ValueError('Acoustic head feature shape mismatch')
        self.np = np
        self.labels = sorted(LABELS, key=LABELS.get)
        self.sha = settings['head_sha256']

    def predict(self, path, seconds):
        blob = path.read_bytes()
        if len(blob) < 20 or blob[:4] != b'WAFT':
            raise ValueError('Missing acoustic feature header')
        layers, frames, states, offset = struct.unpack('<4I', blob[4:20])
        if ((layers, frames, states) != self.shape or offset != 0
                or len(blob) != 20 + 4*math.prod(self.shape)):
            raise ValueError('Acoustic features have another shape or encoder offset')
        features = self.np.frombuffer(blob[20:], dtype='<f4').reshape(self.shape).copy()
        if not self.np.isfinite(features).all():
            raise ValueError('Nonfinite acoustic features')
        began = time.perf_counter()
        padded = self.session.run(None, {'encoder_layers': features})[0]
        head_s = time.perf_counter()-began
        # Preserve released ten-second head semantics; never label 30 s padding.
        valid = padded[:math.ceil(seconds/10)]
        if valid.shape != (math.ceil(seconds/10), 527) or not self.np.isfinite(valid).all():
            raise ValueError('Invalid acoustic logits')
        maximum = valid.max(axis=0)
        indices = self.np.argsort(-maximum, kind='stable')[:2]
        record = dict(schema='sound-event/1', span_seconds=[0, seconds],
                      audio_available_seconds=seconds, nominal_window_seconds=10,
                      score_semantics='raw_uncalibrated_logits',
                      candidates=[dict(label=self.labels[i], class_index=int(i), logit=float(maximum[i]))
                                  for i in indices])
        validate(record)
        return record, head_s


class Adapter:
    def __init__(self, settings, feature_path, process):
        self.settings, self.features, self.process = settings, feature_path, process
        self.head = AcousticHead(settings)
        self.lock = threading.Lock()
        self.serial = 0

    def infer(self, body, content_type, tagged):
        seconds = audio_seconds(body, content_type) if tagged else None
        with self.lock:
            self.serial += 1
            request_id = self.serial
            self.features.unlink(missing_ok=True)
            began = time.perf_counter()
            req = urllib.request.Request(self.settings['backend_url']+'/inference', data=body,
                                         headers={'Content-Type': content_type})
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()
                media = response.headers.get('Content-Type', 'text/plain')
            asr_s = time.perf_counter()-began
            if not tagged:
                return raw, media
            record, error, head_s = None, None, 0
            began = time.perf_counter()
            try:
                record, head_s = self.head.predict(self.features, seconds)
                record.update(request_id=request_id, source='whisper.cpp-shared-encoder/Whisper-AT',
                              head_sha256=self.head.sha)
            except Exception as exc:
                # Metadata failure does not invalidate the already admitted ASR.
                error = str(exc)[:300]
            finally:
                self.features.unlink(missing_ok=True)
            payload = dict(vtt=raw.decode('utf-8'), sound_event=record, sound_error=error,
                           timings=dict(asr_s=asr_s, acoustic_head_s=head_s,
                                        acoustic_total_s=time.perf_counter()-began), request_id=request_id)
            return json.dumps(payload, separators=(',', ':')).encode(), 'application/json'


def handler(adapter):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path != '/' or adapter.process.poll() is not None:
                self.send_error(503, 'Whisper backend unavailable')
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'Whisper acoustic adapter ready')

        def do_POST(self):
            if self.path != '/inference':
                self.send_error(404)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2*1024*1024:
                    raise ValueError('Invalid request length')
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError('Truncated request')
                raw, media = adapter.infer(body, self.headers.get('Content-Type', ''),
                                           self.headers.get('X-Voice-Sound') == '1')
            except ValueError as exc:
                self.send_error(400, str(exc))
                return
            except (OSError, RuntimeError) as exc:
                self.send_error(502, str(exc))
                return
            self.send_response(200)
            self.send_header('Content-Type', media)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    return Handler


def bind_parent_lifetime():
    # Linux nodes terminate owned workers after a five-second grace period.
    # Ensure a forcibly killed adapter cannot orphan its private GPU backend.
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
        raise OSError('Cannot bind private Whisper lifetime to its adapter')
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    settings = cfg['acoustic_input']
    if settings['mode'] != 'whisper_at_v2':
        raise ValueError('Unknown acoustic adapter mode')
    library = settings['library']
    if digest(library) != settings['library_sha256']:
        raise ValueError('Whisper feature tap hash mismatch')
    host, port = settings['listen_host'], settings['listen_port']
    if host != '127.0.0.1' or not settings['backend_url'].startswith('http://127.0.0.1:'):
        raise ValueError('The shared-feature adapter requires private loopback endpoints')
    # Refuse an occupied backend before starting a child: a stale server must
    # never silently supply another model's text/features to this adapter.
    from urllib.parse import urlparse
    address = urlparse(settings['backend_url'])
    with socket.socket() as check:
        if check.connect_ex((address.hostname, address.port)) == 0:
            raise ValueError('Private Whisper backend port is occupied')
    native = settings['native_command']
    if (native[native.index('--port')+1] != str(address.port)
            or native[native.index('--host')+1] != '127.0.0.1'):
        raise ValueError('Private native port mismatch')
    directory = Path(cfg['runtime_dir'])/'acoustic'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix='request-', dir=directory) as temp:
        features = Path(temp)/'encoder.bin'
        env = os.environ.copy()
        env['WHISPER_AT_FEATURE_PATH'] = str(features)
        env['LD_LIBRARY_PATH'] = str(Path(library).parent) + ':' + settings['native_library_dir']
        process = subprocess.Popen(native, env=env,
                                   **({'preexec_fn': bind_parent_lifetime} if os.name == 'posix' else {}))
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        try:
            adapter = Adapter(settings, features, process)
            deadline = time.monotonic()+90
            while not stop.is_set():
                if process.poll() is not None:
                    raise RuntimeError('Private Whisper backend exited')
                try:
                    with urllib.request.urlopen(settings['backend_url']+'/', timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Private Whisper backend startup timed out')
                    stop.wait(.1)
            if stop.is_set():
                return
            server = ThreadingHTTPServer((host, port), handler(adapter))
            server.daemon_threads = True
            server.timeout = .2
            print('Acoustic adapter ready; shared encoder, CPU head, request serialization', flush=True)
            try:
                while not stop.is_set():
                    if process.poll() is not None:
                        raise RuntimeError('Private Whisper backend exited')
                    server.handle_request()
            finally:
                server.server_close()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)


if __name__ == '__main__':
    main()
