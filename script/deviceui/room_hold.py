"""Reversible, word-boundary playback hold for an uncertain room collision."""
import threading
import time


def word_ends(schedule, base, rate):
    """Piper schedule -> absolute mono sample positions at ends of words."""
    marks = []
    last_end = 0.0
    for line in schedule.splitlines():
        try:
            start, duration, phoneme = line.split('\t', 2)
            start, duration = float(start), float(duration)
        except (ValueError, TypeError):
            continue
        if duration < 0 or start < 0: continue
        last_end = max(last_end, start+duration)
        if phoneme.isspace() and start > 0:
            marks.append(base+round(start*rate/1000))
    if last_end: marks.append(base+round(last_end*rate/1000))
    return sorted({mark for mark in marks if mark > base})


class WindowsWordOutput:
    """Gate WASAPI writes at a Piper word end while capture continues."""
    def __init__(self, stream, yielded, rate=22050, grace_ms=0):
        self.stream, self.yielded = stream, yielded
        self.latency = stream.latency
        self.lock = threading.Condition()
        self.boundaries = []
        self.position = 0
        self.submitted = 0
        self.grace_samples = round(rate*grace_ms/1000)
        self.hold_not_before = None
        self.requested = self.held = False
        self.last_written = 0
        self.held_at = None
        self.held_samples = None

    def start(self): return self.stream.start()
    def stop(self): return self.stream.stop()
    def abort(self):
        self.resume()
        return self.stream.abort()
    def close(self): return self.stream.close()

    def add_boundaries(self, marks):
        with self.lock:
            self.boundaries.extend(mark for mark in marks if mark > self.position)
            self.boundaries = sorted(set(self.boundaries))[-256:]
            self.lock.notify_all()

    def hold(self):
        with self.lock:
            if not self.requested:
                self.held_at = None
                self.held_samples = None
                # Include a write already in flight. It cannot be taken back;
                # repeated metadata/hold notifications must not move this gate.
                self.hold_not_before = self.submitted+self.grace_samples
            self.requested = True
            self.lock.notify_all()

    def resume(self):
        with self.lock:
            self.requested = self.held = False
            self.hold_not_before = None
            self.held_at = None
            self.held_samples = None
            self.lock.notify_all()

    def write(self, stereo):
        # Each stereo frame corresponds to one mono sample in the alignment.
        view = memoryview(stereo)
        offset = 0
        self.last_written = 0
        underflow = False
        while offset < len(view) and not self.yielded.is_set():
            with self.lock:
                self.boundaries = [b for b in self.boundaries if b >= self.position]
                if (self.requested and self.position >= self.hold_not_before
                        and self.position in self.boundaries):
                    if not self.held:
                        self.held = True
                        self.held_at = time.time()
                        self.held_samples = self.position
                while self.held and self.requested and not self.yielded.is_set():
                    self.lock.wait(.02)
                if self.yielded.is_set(): break
                end = self.position+(len(view)-offset)//4
                if self.requested:
                    boundary = next((b for b in self.boundaries
                                     if b > self.position and b >= self.hold_not_before), None)
                    if boundary is not None: end = min(end, boundary)
                count = end-self.position
                self.submitted = end
            if count <= 0: break
            underflow = bool(self.stream.write(view[offset:offset+count*4])) or underflow
            offset += count*4
            self.last_written += count
            with self.lock: self.position += count
        return underflow
