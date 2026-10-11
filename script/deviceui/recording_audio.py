"""Session-owned speaker-output capture and digital recording mix.

No microphone, synthesis queue, or conversation control is touched. Windows
requires PyAudioWPatch==0.2.12.8; Linux uses the configured Pulse sink monitor
or the audio authority's final playback-block tap.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import queue
import re
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request

from recording import command

# Cortex's Pulse monitor delivers bursts up to ~90 ms apart. This is a
# recording-only jitter allowance, not latency added to physical speech.
RECORDING_BUFFER_MS = 200


def monitor_rate(sink):
    sources = json.loads(command(['pactl', '-f', 'json', 'list', 'sources']))
    matches = [s for s in sources if s['name'] == sink+'.monitor']
    if len(matches) != 1: raise ValueError('Configured speaker monitor unavailable or ambiguous')
    match = re.search(r'\b(\d+)Hz$', matches[0]['sample_specification'])
    if not match: raise ValueError('Unknown speaker monitor sample rate')
    rate = int(match[1])
    if rate not in (16000, 22050, 24000, 44100, 48000):
        raise ValueError('Unsupported speaker monitor sample rate')
    return rate


class PlaybackCapture:
    """One bounded, authenticated HTTP subscriber to this node's actual output."""
    def __init__(self, sink, playback_socket=None):
        self.sink = sink
        self.playback_socket = playback_socket
        self.rate, self.channels = 48000, 2
        self.chunks = queue.Queue(maxsize=15)  # 300 ms; never accumulate stale speech.
        self.cancel = threading.Event()
        self.ready = threading.Event()
        self.error = ''
        self.proc = None
        self.thread = threading.Thread(target=self.work, daemon=True)

    def put(self, data):
        if self.cancel.is_set(): return
        try: self.chunks.put_nowait(data)
        except queue.Full:
            self.error = 'Digital capture consumer fell behind'
            self.cancel.set()
        self.ready.set()

    def start(self):
        self.thread.start()
        if not self.ready.wait(5):
            self.close()
            raise RuntimeError('No PCM from the configured speaker output')
        if self.error:
            self.close()
            raise RuntimeError(self.error)
        return self

    def work(self):
        try:
            if sys.platform == 'win32':
                self.windows()
            elif self.playback_socket:
                self.rendered()
            else:
                # Preserve the endpoint clock/rate, as the classroom relay did.
                # Cortex's reSpeaker monitor is 16 kHz; resample once at the mix.
                self.rate = monitor_rate(self.sink)
                size = self.rate//50*self.channels*2
                self.proc = subprocess.Popen(['parec', '--raw', '--device='+self.sink+'.monitor',
                    '--format=s16le', '--rate='+str(self.rate), '--channels=2', '--latency-msec=20',
                    '--process-time-msec=10'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                while not self.cancel.is_set():
                    data = self.proc.stdout.read(size)  # 20 ms, stereo s16le.
                    if len(data) != size: raise RuntimeError('Speaker monitor capture stopped')
                    self.put(data)
        except Exception as exc:
            if not self.cancel.is_set(): self.error = str(exc)
        finally:
            if self.proc:
                if self.proc.poll() is None: self.proc.terminate()
                try: self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait(timeout=2)
                self.proc.stdout.close()
            self.ready.set()

    def rendered(self):
        """Final card-clocked blocks, never queued or canceled synthesis.

        Some Pulse/ALSA monitor captures contain holes absent from physical
        playback. The audio authority's output tap avoids that monitor path.
        Its nonblocking sender closes on backpressure; it never drops PCM.
        """
        self.rate, self.channels = 16000, 1
        header = struct.Struct('<BBHHI')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(1)
            stream.connect(self.playback_socket)
            stream.sendall(header.pack(0x01, ord('o'), 0, 0, 0))
            def read(size):
                data = bytearray()
                while len(data) < size:
                    if self.cancel.is_set(): raise RuntimeError('Recording capture canceled')
                    chunk = stream.recv(size-len(data))
                    if not chunk: raise RuntimeError('Rendered playback capture stopped')
                    data.extend(chunk)
                return bytes(data)
            while not self.cancel.is_set():
                data = bytearray()
                for _ in range(2):  # Keep the HTTP queue's 20 ms chunk contract.
                    magic, kind, w, h, size = header.unpack(read(header.size))
                    if (magic, kind, w, h, size) != (0x01, ord('P'), 0, 0, 320):
                        raise RuntimeError('Invalid rendered playback frame; rebuild far-field-service')
                    data.extend(read(size))
                self.put(bytes(data))

    def windows(self):
        import pyaudiowpatch as pa
        with pa.PyAudio() as audio, ExitStack() as stack:
            devices = [d for d in audio.get_loopback_device_info_generator()
                       if d['name'].removesuffix(' [Loopback]') == self.sink]
            if len(devices) != 1: raise ValueError('Configured speaker loopback unavailable or ambiguous')
            device = devices[0]
            self.rate = int(device['defaultSampleRate'])
            self.channels = int(device['maxInputChannels'])
            def capture(data, frames, timing, status):
                if status:
                    self.error = 'Windows speaker loopback overflow or discontinuity'
                    self.cancel.set()
                self.put(data)
                return (None, pa.paComplete if self.cancel.is_set() else pa.paContinue)
            stream = stack.enter_context(audio.open(format=pa.paInt16, channels=self.channels,
                rate=self.rate, input=True, input_device_index=device['index'],
                frames_per_buffer=self.rate//50, stream_callback=capture))
            # WASAPI loopback must maintain a continuous clock during silence.
            # A zero-valued render stream keeps the endpoint clock running even
            # when Piper has closed its stream between replies. It plays no sound.
            outputs = [d for d in audio.get_device_info_generator()
                       if d['hostApi'] == device['hostApi'] and d['name'] == self.sink
                       and d['maxOutputChannels'] >= self.channels]
            if len(outputs) != 1: raise ValueError('Configured speaker output unavailable or ambiguous')
            stack.enter_context(audio.open(format=pa.paInt16, channels=self.channels, rate=self.rate,
                output=True, output_device_index=outputs[0]['index'], frames_per_buffer=self.rate//50,
                stream_callback=lambda data, frames, timing, status: (bytes(frames*self.channels*2), pa.paContinue)))
            while not self.cancel.wait(.1):
                if not stream.is_active(): raise RuntimeError('Windows speaker loopback stopped')

    def __iter__(self):
        try:
            while not self.cancel.is_set():
                if self.error: raise RuntimeError(self.error)
                try: yield self.chunks.get(timeout=.5)
                except queue.Empty:
                    if not self.thread.is_alive(): raise RuntimeError(self.error or 'Speaker capture stopped')
            if self.error: raise RuntimeError(self.error)
        finally:
            self.close()

    def close(self):
        self.cancel.set()
        if self.proc and self.proc.poll() is None: self.proc.terminate()
        if self.thread.ident: self.thread.join(timeout=5)


class DigitalRelay:
    def __init__(self, node, sink, log_path=None):
        self.node, self.sink = node, sink
        self.log_path = log_path
        self.cancel = threading.Event()
        self.ready = threading.Event()
        self.error = ''
        self.player = None
        self.thread = threading.Thread(target=self.work, daemon=True)

    def work(self):
        try:
            req = urllib.request.Request(self.node['url'].rstrip('/')+'/api/recording/pcm',
                headers={'X-Voice-Key': self.node['key']})
            with ExitStack() as resources:
                source = resources.enter_context(urllib.request.urlopen(req, timeout=4))
                rate = int(source.headers.get('X-PCM-Rate', 0))
                channels = int(source.headers.get('X-PCM-Channels', 0))
                if (source.headers.get('X-PCM-Format') != 's16le' or
                        rate not in (16000, 22050, 24000, 44100, 48000) or channels not in (1, 2)):
                    raise ValueError('Unsupported speaker capture format')
                if self.cancel.is_set(): return
                log = resources.enter_context(Path(self.log_path).open('wb')) if self.log_path else subprocess.DEVNULL
                self.player = subprocess.Popen(['pacat', '-v', '--raw', '--playback', '--device='+self.sink,
                    '--format=s16le', '--rate='+str(rate), '--channels='+str(channels),
                    '--latency-msec='+str(RECORDING_BUFFER_MS), '--process-time-msec=10', '--volume=58982',
                    '--client-name=voice-trio-recording', '--stream-name='+self.node['id']],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                size = rate//50*channels*2
                received = 0
                while not self.cancel.is_set():
                    data = source.read(size)
                    if len(data) != size: raise RuntimeError('Speaker capture connection closed')
                    if self.player.poll() is not None: raise RuntimeError('Digital mix playback stopped')
                    self.player.stdin.write(data)
                    self.player.stdin.flush()
                    received += len(data)
                    # Fill the recording buffer with real PCM (including idle
                    # silence) before the session opening can be released.
                    if received >= rate*channels*2*RECORDING_BUFFER_MS//1000:
                        self.ready.set()
        except Exception as exc:
            if not self.cancel.is_set(): self.error = self.node['id']+': '+str(exc)
        finally:
            self.ready.set()
            self.stop_player()

    def stop_player(self):
        if self.player and self.player.poll() is None:
            self.player.terminate()
            try: self.player.wait(timeout=2)
            except subprocess.TimeoutExpired: self.player.kill(); self.player.wait(timeout=2)
        if self.player and self.player.stdin: self.player.stdin.close()

    def check(self):
        if self.error: raise RuntimeError('Digital recording feed failed: '+self.error)
        if self.ready.is_set() and not self.thread.is_alive() and not self.cancel.is_set():
            raise RuntimeError('Digital recording feed stopped: '+self.node['id'])


class DigitalMix:
    def __init__(self, nodes):
        self.nodes = list(nodes)
        self.relays = []
        self.module = None
        self.sink = 'voice_trio_digital_'+secrets.token_hex(6)

    def start(self, cancel, log_dir=None):
        if log_dir: Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.module = command(['pactl', 'load-module', 'module-null-sink',
            'sink_name='+self.sink, 'rate=48000', 'channels=2', 'channel_map=front-left,front-right'])
        for node in self.nodes:
            relay = DigitalRelay(node, self.sink, Path(log_dir)/(node['id']+'.log') if log_dir else None)
            self.relays.append(relay)
            relay.thread.start()
        deadline = time.monotonic()+10
        while not cancel.wait(.05):
            self.check()
            if all(r.ready.is_set() for r in self.relays): return self.sink+'.monitor'
            if time.monotonic() >= deadline: raise RuntimeError('Digital recording feeds did not become ready')
        return self.sink+'.monitor'  # Recording owns cancellation and cleanup.

    def check(self):
        for relay in self.relays: relay.check()

    def close(self):
        for relay in self.relays:
            relay.cancel.set()
            if relay.player and relay.player.poll() is None: relay.player.terminate()
        for relay in self.relays: relay.thread.join(timeout=5)
        if self.module:
            command(['pactl', 'unload-module', self.module])
            self.module = None
