"""Incremental Whisper windows and immutable, word-boundary ASR commits.

The capture thread owns this state. Inference runs separately against immutable
snapshots; applying its result never removes audio captured in the meantime.
"""
from collections import deque
import re


def words(text):
    # Captions are not spoken words. Preserve spelling (including unknown names).
    return re.sub(r'\[[^]]*\]|\([^)]*\)|[♪♫]', ' ', text).split()


def asr_prompt(name_hint, context_tail):
    """Keep learned spelling near the short transcript tail on each ASR pass."""
    return (name_hint+' '+context_tail[-250:]).strip() if name_hint else context_tail


def parse_vtt(text):
    result, segments, cue, lines = [], [], None, []
    def flush():
        if cue is not None:
            result.extend(words(' '.join(lines)))
            segments.append((cue, len(result)))
    for line in text.splitlines():
        match = re.search(r'-->\s*(\d+):(\d+):(\d+)[.,](\d+)', line)
        if match:
            flush()
            h, m, s, ms = map(int, match.groups())
            cue, lines = round((h*3600+m*60+s+ms/1000)*16000), []
        elif cue is not None and line.strip():
            lines.append(line)
    flush()
    if not segments and text.strip() not in ('', 'WEBVTT'):
        raise ValueError('Whisper did not return VTT segment timestamps')
    return dict(words=result, segments=segments)


class StreamingUtterance:
    def __init__(self, level=200, silence=.8, commit=1.1, settle=.3, window=30):
        self.level, self.silence, self.commit, self.settle, self.window = level, silence, commit, settle, window
        self.preroll = deque(maxlen=10)
        self.audio = bytearray()
        self.total = self.base = self.last_voice = self.last_pass = self.covered_voice = 0
        self.quiet = self.voiced = 0
        self.previous = self.final_words = []
        self.committed = []
        self.local_committed = 0
        self.active = False
        self.last_elapsed = None
        self.last_mode = None
        self.passes = 0
        self.sound_event = None
        self.sound_timings = None
        self.sound_error = None

    def feed(self, pcm, rms):
        voiced = rms >= self.level
        if not self.active:
            self.preroll.append(pcm)
            if not voiced:
                return
            self.active = True
            self.audio.extend(b''.join(self.preroll))
            self.total = len(self.audio)//2
        else:
            self.audio.extend(pcm)
            self.total += len(pcm)//2
        duration = len(pcm)/32000
        self.voiced += duration if voiced else 0
        self.quiet = 0 if voiced else self.quiet+duration
        if voiced:
            self.last_voice = self.total
        if len(self.audio) > self.window*32000:
            raise RuntimeError('ASR unconfirmed window exceeded 30 seconds; stopping without discarding committed input')

    def request(self, held):
        if not self.active or self.voiced < .2:
            return None
        mode = ('final' if self.quiet >= self.silence and not held else
                'settle' if self.quiet >= self.settle else 'partial')
        if mode == 'partial':
            if self.total-self.last_pass < self.commit*16000:
                return None
        elif self.settled():
            return None
        self.last_pass = self.total
        return dict(pcm=bytes(self.audio), end=self.total, base=self.base, mode=mode,
                    prompt=' '.join(self.committed[:-self.local_committed] if self.local_committed else self.committed)[-400:])

    def apply(self, result, request):
        if request['base'] != self.base:
            raise RuntimeError('ASR window changed while inference was pending')
        current = result['words']
        if result.get('sound_event') is not None:
            from sound_event import align
            try:
                self.sound_event = align(result['sound_event'], request)
                self.sound_error = None
            except (ValueError, KeyError, TypeError) as exc:
                self.sound_error = str(exc)
        self.sound_timings = result.get('sound_timings', self.sound_timings)
        self.sound_error = result.get('sound_error') or self.sound_error
        self.last_elapsed, self.last_mode = result.get('elapsed'), request['mode']
        self.passes += 1
        agree = 0
        for a, b in zip(self.previous, current):
            if a != b:
                break
            agree += 1
        # Hold the newest word, which may still be acoustically incomplete.
        agree = min(agree, max(0, len(current)-1))
        added = current[self.local_committed:agree] if agree > self.local_committed else []
        piece = (' ' if self.committed else '')+' '.join(added) if added else ''
        self.committed.extend(added)
        self.local_committed += len(added)
        self.previous = list(current)
        self.final_words = list(current)
        self.covered_voice = request['end']
        # Trim only complete, committed segments with at least 250ms beyond
        # their end. Never move a segment boundary merely to keep overlap.
        cut = count = 0
        for end, nwords in result['segments']:
            if 0 < nwords <= self.local_committed and end <= len(request['pcm'])//2-4000:
                cut, count = end, nwords
        if cut > 0:
            del self.audio[:cut*2]
            self.base += cut
            self.local_committed -= count
            self.final_words = current[count:]
            self.previous = []
        return piece

    def ready(self, held):
        return (self.active and not held and self.quiet >= self.silence
                and (self.voiced < .2 or self.settled()))

    def settled(self):
        # A partial pass ending exactly at the last voiced block may contain
        # a clipped final word. Require a pass with trailing silence; reuse it
        # during endpoint holds only while no newer speech has arrived.
        return self.last_mode in ('settle', 'final') and self.covered_voice >= self.last_voice

    def finish(self):
        tail_words = self.final_words[self.local_committed:]
        text = ' '.join(self.committed+tail_words)
        tail = (' ' if self.committed and tail_words else '')+' '.join(tail_words)
        return text, tail
