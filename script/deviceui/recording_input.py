"""Nonblocking tap of the exact processed PCM entering the room ASR logic.

The HTTP consumer owns serialization. The capture thread only enqueues bytes;
backpressure ends the diagnostic stream and never stalls conversation audio.
"""
import json
import queue
import secrets
import struct
import threading
import time

FRAME = struct.Struct('<cI')
BLOCK = struct.Struct('<QdB')  # WAV sample offset, node capture monotonic time, eligible for ASR
RATE = 16000
FORMAT = 'little-gemma-asr-input-v1'


def frame(kind, payload):
    return FRAME.pack(kind, len(payload))+payload


class InputSubscription:
    def __init__(self, owner, speaker):
        self.owner = owner
        self.id = secrets.token_hex(16)
        self.lock = threading.Lock()
        self.chunks = queue.Queue(maxsize=100)  # Two seconds at the existing 20 ms capture cadence.
        self.ready = threading.Event()
        self.ended = threading.Event()
        self.samples = 0
        self.error = ''
        self.reason = ''
        self.metadata = dict(session=owner.session, speaker_id=speaker['speaker_id'],
                             speaker_name=speaker['display_name'], protocol=FORMAT, capture_id=self.id,
                             rate=RATE, channels=1, sample_format='s16le',
                             node_unix=time.time(), node_monotonic=time.monotonic(),
                             stage='processed continuous input before VAD and ASR windowing')

    def put(self, captured, pcm, eligible):
        with self.lock:
            if self.ended.is_set(): return
            if not pcm or len(pcm) % 2:
                self._finish('invalid-pcm', 'Invalid ASR input PCM block')
                return
            try:
                self.chunks.put_nowait((self.samples, captured, bool(eligible), pcm))
            except queue.Full:
                self._finish('overflow', 'ASR recording consumer fell behind; track is incomplete')
                return
            self.samples += len(pcm)//2
            self.ready.set()

    def _finish(self, reason, error):
        if self.ended.is_set(): return
        self.reason, self.error = reason, error
        self.ended.set()
        self.ready.set()

    def finish(self, reason='recording-stopped', error=''):
        with self.lock: self._finish(reason, error)

    def summary(self):
        with self.lock:
            return dict(samples=self.samples, reason=self.reason, error=self.error)

    def close(self):
        self.finish('consumer-closed')
        self.owner.release(self)

    def __iter__(self):
        try:
            yield frame(b'M', json.dumps(self.metadata).encode())
            while True:
                try: offset, captured, eligible, pcm = self.chunks.get(timeout=.1)
                except queue.Empty:
                    if self.ended.is_set(): break
                    continue
                yield frame(b'P', BLOCK.pack(offset, captured, eligible)+pcm)
            yield frame(b'E', json.dumps(self.summary()).encode())
        finally:
            self.close()


class InputTap:
    def __init__(self, session):
        self.session = session
        self.lock = threading.Lock()
        self.current = None
        self.closed = False

    def subscribe(self, speaker):
        with self.lock:
            if self.closed: raise ValueError('Room audio input is stopped')
            if self.current is not None: raise ValueError('ASR input is already being recorded')
            self.current = InputSubscription(self, speaker)
            return self.current

    def feed(self, captured, pcm, eligible):
        current = self.current
        if current is not None: current.put(captured, pcm, eligible)

    def stop(self, capture_id):
        with self.lock:
            current = self.current
            if current is not None:
                if current.id != capture_id: raise ValueError('Stale ASR recording')
                current.finish()
            return dict(ok=True)

    def release(self, subscription):
        with self.lock:
            if self.current is subscription: self.current = None

    def close(self, error=''):
        with self.lock:
            self.closed = True
            if self.current is not None:
                self.current.finish('pipeline-stopped', error)
