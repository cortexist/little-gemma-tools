"""Parallel CAM++ windows with the classroom's 500/250 ms confirmation cadence.

The CPU extractor lives in a separate process: its Python binding cannot hold
the capture/Whisper/prefill interpreter's GIL. Only listener-local tracks compete.
Names are learned later from completed transcripts, never from a floor grant.
"""
import json
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ParallelIdentity:
    def __init__(self, model, sha256, log):
        self.jobs = queue.Queue(maxsize=3)
        self.stopped = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), model, sha256],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        self.thread = threading.Thread(target=self.run, name='room-identity', daemon=True)
        self.thread.start()
        if not self.ready.wait(30) or self.error:
            self.close()
            raise RuntimeError(self.error or 'Speaker identity did not become ready')

    def submit(self, ticket, job):
        if self.error: raise RuntimeError(self.error)
        try: self.jobs.put_nowait((ticket, job))
        except queue.Full:
            raise RuntimeError('Speaker identification fell behind; refusing unattributed turn decisions')

    def run(self):
        try:
            if self.proc.stdout.readline() != b'ready\n':
                raise RuntimeError('Speaker identity failed to load')
            self.ready.set()
            while not self.stopped.is_set():
                try: ticket, job = self.jobs.get(timeout=.1)
                except queue.Empty: continue
                if job is None:
                    ticket.done.set()
                    continue
                if ticket.cancelled: continue
                pcm = job.pop('pcm')
                self.proc.stdin.write(struct.pack('<I', len(pcm))+pcm)
                self.proc.stdin.flush()
                result = json.loads(self.proc.stdout.readline())
                if 'error' in result: raise RuntimeError(result['error'])
                ticket.result(result['embedding'], job, result['compute_s'])
        except Exception as exc:
            if not self.stopped.is_set(): self.error = 'Speaker identity: '+str(exc)
        finally:
            self.ready.set()

    def close(self):
        self.stopped.set()
        if self.proc.poll() is None: self.proc.terminate()
        try: self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill(); self.proc.wait(timeout=5)
        self.thread.join(timeout=5)
        self.proc.stdin.close(); self.proc.stdout.close()


class LiveUtterance:
    """One ASR utterance, possibly containing several acoustic speech segments.

Confirmed changes invalidate single-speaker attribution. This detects sequential
changes; it does not claim simultaneous-overlap detection from CAM++ embeddings.
"""
    def __init__(self, worker, session, lock, publish, level=200):
        self.worker, self.session, self.lock, self.publish = worker, session, lock, publish
        self.level = level
        self.done = threading.Event()
        self.cancelled = False
        self.started = time.monotonic()
        self.audio = bytearray()
        self.samples = self.next_match = self.quiet = 0
        self.segment = 0
        self.active = False
        self.candidate = None
        self.count = 0
        self.candidate_vector = None
        self.voices = set()
        self.unresolved = set()
        self.vectors = []
        self.windows = []
        self.first_match_s = None
        self.closed = False
        self.enrollment_audio = bytearray()
        self.enrollment_size = 0
        self.enrollment_voice = 0
        self.last_voice = 0
        self.next_enrollment = 96000
        self.enrollment = None
        self.enrollment_timings = []
        self.segment_audio = bytearray()
        self.segment_checks = []
        self.checked_segments = set()

    def feed(self, pcm, rms):
        self.enrollment_audio.extend(pcm)
        del self.enrollment_audio[256000:]
        size = len(self.enrollment_audio)
        speech = rms >= self.level
        if speech: self.last_voice = size
        if size >= self.next_enrollment:
            self.checkpoint_enrollment()
            self.next_enrollment = min(self.next_enrollment+96000, 256000) if self.next_enrollment < 256000 else 256001
        if speech and not self.active:
            self.segment += 1
            self.audio.clear()
            self.segment_audio.clear()
            self.samples, self.next_match, self.quiet = 0, 8000, 0
            self.active = True
            with self.lock:
                self.candidate, self.count = None, 0
                self.publish(dict(status='identifying', track_id=None))
        if not self.active:
            # Very brief speech may reach the minimum enrollment length only
            # after the 350 ms quiet transition, still before ASR admission.
            self.checkpoint_enrollment()
            return
        self.audio.extend(pcm)
        self.segment_audio.extend(pcm)
        del self.segment_audio[:-256000]
        del self.audio[:-64000]  # two seconds, unchanged from classroom
        self.samples += len(pcm)//2
        self.quiet = 0 if speech else self.quiet+len(pcm)//2
        if self.quiet >= 5600:  # 350 ms quiet expires a current identification
            self.active = False
            # Prepare the long embedding during ASR settling/endpoint buffering.
            # Continued silence must not schedule another extraction at admit.
            self.checkpoint_enrollment()
            self.checkpoint_segment()
            with self.lock:
                self.candidate, self.count = None, 0
                self.publish(dict(status='quiet', track_id=None))
        elif speech and self.samples >= self.next_match:
            self.next_match = self.samples+4000
            self.worker.submit(self, dict(pcm=bytes(self.audio), segment=self.segment,
                end_s=time.monotonic()-self.started, submitted=time.monotonic(),
                duration=len(self.audio)/32000, kind='live'))

    def submit_enrollment(self):
        self.enrollment_size = len(self.enrollment_audio)
        self.enrollment_voice = self.last_voice
        self.worker.submit(self, dict(pcm=bytes(self.enrollment_audio), kind='enrollment',
            duration=self.enrollment_size/32000, submitted=time.monotonic(),
            segment=self.segment, end_s=time.monotonic()-self.started))

    def checkpoint_enrollment(self):
        if (len(self.enrollment_audio) >= 24000 and len(self.enrollment_audio) > self.enrollment_size
                and self.last_voice > self.enrollment_voice):
            self.submit_enrollment()

    def checkpoint_segment(self):
        # Weak prefixes after a pause are not proof of another speaker. Resolve
        # them using this segment alone, never the preceding speaker's audio.
        # Still run in the CPU worker during the quiet/ASR endpoint buffer.
        with self.lock:
            if (self.segment not in self.unresolved or self.segment in self.checked_segments
                    or len(self.segment_audio) < 24000):
                return
            self.checked_segments.add(self.segment)
            self.worker.submit(self, dict(pcm=bytes(self.segment_audio), kind='segment',
                segment=self.segment, duration=len(self.segment_audio)/32000,
                submitted=time.monotonic(), end_s=time.monotonic()-self.started))

    def result(self, embedding, job, compute_s):
        with self.lock:
            if self.cancelled: return
            match = self.session.preview(embedding)
            if job.get('kind') == 'enrollment':
                # Longer windows establish new tracks. Do not enroll from the
                # first phoneme: a 750 ms centroid can fragment a single voice.
                self.enrollment = (embedding, match)
                self.enrollment_timings.append(dict(duration=job['duration'], compute_s=compute_s,
                    processing_s=time.monotonic()-job['submitted'], match=match))
                return
            if job.get('kind') == 'segment':
                self.segment_checks.append(dict(segment=job['segment'], duration=job['duration'],
                    compute_s=compute_s, processing_s=time.monotonic()-job['submitted'], match=match))
                if match['status'] == 'matched':
                    self.unresolved.discard(job['segment'])
                    self.voices.add(match['track_id'])
                return
            candidate = match['track_id'] or match['status']
            same = self.candidate == (job['segment'], candidate)
            if candidate == 'unknown' and same:
                same = sum(a*b for a,b in zip(embedding, self.candidate_vector)) >= self.session.match_threshold
            self.count = self.count+1 if same else 1
            self.candidate, self.candidate_vector = (job['segment'], candidate), embedding
            confirmed = self.count >= 2 and match['status'] == 'matched'
            if self.count >= 2 and match['status'] == 'unknown' and self.voices:
                self.unresolved.add(job['segment'])
            record = dict(segment=job['segment'], end_s=job['end_s'], duration=job['duration'],
                          compute_s=compute_s, processing_s=time.monotonic()-job['submitted'],
                          status=match['status'], track_id=match['track_id'], similarity=match['similarity'])
            if confirmed:
                self.unresolved.discard(job['segment'])
                self.voices.add(match['track_id'])
                self.vectors.append(embedding)
                self.vectors = self.vectors[-64:]
                if self.first_match_s is None: self.first_match_s = time.monotonic()-self.started
            self.windows.append(record)
            self.windows = self.windows[-400:]
            # A delayed result from an earlier segment must not restore a fresh label.
            if job['segment'] == self.segment and self.active and not self.closed:
                self.publish(dict(match, status='identified' if confirmed else 'unknown' if
                    self.count >= 2 and match['status'] == 'unknown' else 'identifying',
                    first_match_s=self.first_match_s, window_s=job['duration'], compute_s=compute_s))

    def seal(self):
        self.closed = True
        # A resumed voice invalidates the quiet checkpoint. A forced endpoint
        # must still cover it; never trade current attribution for a fast timer.
        self.checkpoint_enrollment()
        self.checkpoint_segment()
        self.worker.submit(self, None)  # barrier follows this utterance's windows

    def finish(self, cancel, timeout=5):
        deadline = time.monotonic()+timeout
        while not self.done.wait(.02):
            if cancel.is_set(): raise RuntimeError('Identification cancelled')
            if self.worker.error: raise RuntimeError(self.worker.error)
            if time.monotonic() > deadline: raise TimeoutError('Current utterance diarisation is not ready')
        with self.lock:
            embedding, match = self.enrollment or (None, dict(status='uncertain', track_id=None))
            usable = (len(self.voices) <= 1 and not self.unresolved and match['status'] in ('matched', 'unknown')
                      and (not self.voices or self.voices == {match['track_id']}))
            status = ('new_voice' if match['status']=='unknown' else 'single_voice') if usable else (
                     'multiple_voices' if len(self.voices)>1 else 'unattributed')
            return dict(embedding=embedding if usable else None, tracks=[match['track_id']] if usable and match['track_id'] else [],
                        status=status,
                        first_match_s=self.first_match_s, windows=list(self.windows),
                        enrollment=list(self.enrollment_timings), segment_checks=list(self.segment_checks),
                        confirmed_tracks=sorted(self.voices), unresolved_segments=sorted(self.unresolved),
                        enrollment_match=match)


def main():
    from session_replay import AcousticEncoder
    from windows_audio import read_exact
    encoder = AcousticEncoder(sys.argv[1], sys.argv[2])
    sys.stdout.buffer.write(b'ready\n'); sys.stdout.buffer.flush()
    while True:
        header = read_exact(sys.stdin.buffer, 4)
        if not header: break
        size, = struct.unpack('<I', header)
        if not 16000 <= size <= 256000: raise ValueError('Invalid live identity window')
        pcm = read_exact(sys.stdin.buffer, size)
        began = time.monotonic()
        try: result = dict(embedding=encoder.pcm(pcm, minimum=.5), compute_s=time.monotonic()-began)
        except Exception as exc: result = dict(error=str(exc))
        sys.stdout.buffer.write(json.dumps(result).encode()+b'\n'); sys.stdout.buffer.flush()


if __name__ == '__main__': main()
