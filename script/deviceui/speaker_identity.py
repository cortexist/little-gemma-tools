#!/usr/bin/env python3
"""Optional one-peer voice tracker. Speakers are UI estimates, never word attribution."""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import struct
import threading
import time

from speaker_collections import SpeakerCollection

RATE = 16000
HEADER = struct.Struct('<BBHHI')


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class PeerIdentity:
    """Reconfirm each speech segment; preserve the last peer as historical state."""
    def __init__(self, collection, allowed, threshold=.425, margin=.1, confirmations=2):
        if not 0 < threshold <= 1 or not 0 < margin <= 2 or confirmations < 1:
            raise ValueError('Invalid identity policy')
        self.collection = collection
        self.allowed = collection.allowed({'peer_speaker_ids': allowed})
        self.threshold, self.margin, self.confirmations = threshold, margin, confirmations
        self.last_peer = None
        self.reset('waiting')

    def reset(self, status='identifying'):
        self.status = status
        self.candidate = None
        self.count = 0
        self.current = None

    def observe(self, scores, at):
        if not set(self.collection.by_voice).issubset(scores) or len(scores) < 2 or not all(math.isfinite(v) for v in scores.values()):
            self.reset('uncertain')
            return self.snapshot()
        ordered = sorted(scores, key=scores.get, reverse=True)
        winner = ordered[0]
        margin = scores[winner]-scores[ordered[1]] if len(ordered) > 1 else 0
        speaker_id = self.collection.by_voice.get(winner)
        if scores[winner] < self.threshold or margin < self.margin:
            self.reset('uncertain')
        elif speaker_id not in self.allowed:
            self.reset('other_voice')
        else:
            self.count = self.count+1 if self.candidate == speaker_id else 1
            self.candidate = speaker_id
            self.current = None
            self.status = 'identifying'
            if self.count >= self.confirmations:
                self.current = speaker_id
                self.status = 'identified'
                self.last_peer = dict(self.collection.speaker(speaker_id), at=at)
        return self.snapshot()

    def snapshot(self):
        info = self.collection.speaker(self.current) or {}
        return dict(info, collection_id=self.collection.id, speaker_id=self.current,
                    voice_id=info.get('voice_id'), display_name=info.get('display_name'),
                    status=self.status, last_peer=self.last_peer, mode='single_peer',
                    attribution='estimate', overlap_checked=False, allow_named_address=False)


class Encoder:
    def __init__(self, cfg):
        # Bound auxiliary libraries before importing NumPy or loading inference.
        for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[key] = '1'
        import numpy as np
        import sherpa_onnx
        self.np = np
        if file_sha(cfg['model']) != cfg['model_sha256'] or file_sha(cfg['profiles']) != cfg['profiles_sha256']:
            raise ValueError('Speaker model/profile hash mismatch')
        with np.load(cfg['profiles'], allow_pickle=False) as saved:
            self.profiles = saved['profiles']
            self.voices = saved['speakers'].tolist()
        if self.profiles.ndim != 2 or len(self.profiles) != len(self.voices) or not np.isfinite(self.profiles).all():
            raise ValueError('Invalid enrollment profiles')
        if len(set(self.voices)) != len(self.voices) or len(self.voices) < 2:
            raise ValueError('Enrollment must contain distinct competing identities')
        collection = SpeakerCollection(cfg['collection_id'], cfg['collection'])
        if not set(collection.by_voice).issubset(self.voices):
            raise ValueError('Every known speaker needs an enrolled voice; extra competitors may remain unassigned')
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=cfg['model'], num_threads=1, provider='cpu'))

    def scores(self, pcm):
        np = self.np
        x = np.frombuffer(pcm, '<i2').astype(np.float32)/32768
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=RATE, waveform=x)
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        embedding = np.asarray(self.extractor.compute(stream), np.float32)
        norm = np.linalg.norm(embedding)
        if not np.isfinite(embedding).all() or norm <= 0:
            raise ValueError('Invalid speaker embedding')
        values = self.profiles @ (embedding/norm)
        return {name: float(value) for name, value in zip(self.voices, values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audio-socket', required=True)
    parser.add_argument('--events-socket', required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg.get('backend', 'camplus') != 'camplus':
        raise ValueError('This one-peer worker supports the camplus backend; diarization remains a separate experiment')
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    events = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    events.setblocking(False)
    def emit(state):
        event = dict(kind='speaker_identity', text=json.dumps(state), at=time.time())
        try:
            events.sendto(json.dumps(event).encode(), args.events_socket)
        except (BlockingIOError, FileNotFoundError, ConnectionRefusedError):
            pass  # Monitoring cannot hold the capture worker up.
    collection = SpeakerCollection(cfg['collection_id'], cfg['collection'])
    tracker = PeerIdentity(collection, cfg['peer_speaker_ids'], cfg.get('threshold', .425),
                           cfg.get('margin', .1), cfg.get('confirmations', 2))
    emit(tracker.snapshot())
    encoder = Encoder(cfg)
    incoming = queue.Queue(maxsize=32)
    errors = []
    def read_audio():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as source:
                source.settimeout(.2)
                source.connect(args.audio_socket)
                source.sendall(HEADER.pack(1, ord('e'), 0, 0, 0))
                def read_exact(count):
                    data = bytearray()
                    while len(data) < count and not stop.is_set():
                        try:
                            part = source.recv(count-len(data))
                        except socket.timeout:
                            continue
                        if not part:
                            raise EOFError('Microphone tap closed')
                        data.extend(part)
                    return bytes(data)
                last = time.monotonic()
                while not stop.is_set():
                    header = read_exact(HEADER.size)
                    if stop.is_set():
                        break
                    magic, kind, _, _, size = HEADER.unpack(header)
                    if magic != 1 or size > 65536:
                        raise ValueError('Invalid microphone frame')
                    pcm = read_exact(size)
                    if kind != ord('P'):
                        continue
                    if len(pcm) != 320:
                        raise ValueError('Expected 10 ms mono 16 kHz PCM')
                    now = time.monotonic()
                    gap = now-last > .5
                    last = now
                    if incoming.full():
                        while True:
                            try:
                                incoming.get_nowait()
                            except queue.Empty:
                                break
                        gap = True
                    incoming.put_nowait((pcm, gap))
        except Exception as exc:
            if not stop.is_set():
                errors.append(str(exc))
            stop.set()
    reader = threading.Thread(target=read_audio, daemon=True)
    reader.start()
    frames = deque(maxlen=200)
    active = False
    quiet = 0
    samples = 0
    next_match = RATE//2
    last_emit = 0.
    try:
        while not stop.is_set():
            try:
                pcm, gap = incoming.get(timeout=.2)
            except queue.Empty:
                continue
            if gap:
                tracker.reset('audio_gap')
                frames.clear(); active = False; quiet = 0; samples = 0; next_match = RATE//2
                emit(tracker.snapshot())
            rms = float(encoder.np.sqrt(encoder.np.mean(encoder.np.frombuffer(pcm, '<i2').astype(float)**2)))
            speech = rms >= cfg.get('vad_level', 200)
            if speech and not active:
                tracker.reset('identifying')
                frames.clear(); samples = 0; next_match = RATE//2; active = True
            if active:
                frames.append(pcm)
                samples += 160
                quiet = 0 if speech else quiet+160
                if quiet >= round(.35*RATE):
                    active = False; tracker.reset('quiet'); frames.clear()
                elif speech and samples >= next_match:
                    scores = encoder.scores(b''.join(frames))
                    if scores is not None:
                        tracker.observe(scores, time.time())
                    next_match = samples+RATE//4
            now = time.monotonic()
            if now-last_emit >= .25:
                emit(tracker.snapshot()); last_emit = now
    finally:
        stop.set(); reader.join(timeout=1)
        tracker.reset('unavailable' if errors else 'stopped')
        result = tracker.snapshot()
        if errors:
            result['error'] = errors[0]
        emit(result)
        events.close()
    if errors:
        raise RuntimeError(errors[0])


if __name__ == '__main__':
    main()
