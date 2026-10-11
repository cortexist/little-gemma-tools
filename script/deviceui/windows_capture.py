"""Bounded PCM endpointing and local Whisper requests for the Windows node."""
from collections import deque
import io
import json
import re
import urllib.request
import uuid
import wave


def select_input(sd, name, hostapi='Windows WASAPI', channels=2):
    apis = sd.query_hostapis()
    matches = [i for i, d in enumerate(sd.query_devices())
               if d['name'] == name and d['max_input_channels'] >= channels
               and apis[d['hostapi']]['name'] == hostapi]
    if len(matches) != 1:
        raise ValueError(f'Expected one {hostapi} input named {name!r}; found {len(matches)}')
    return matches[0]


class CapturePCM:
    """Processed channel 0; stateful anti-aliased 48-to-16 kHz conversion."""
    def __init__(self, rate, channels=2):
        import numpy as np
        if rate not in (16000, 48000):
            raise ValueError('Windows capture requires a 16 or 48 kHz endpoint')
        self.rate = rate
        self.channels = channels
        x = np.arange(63)-31
        self.taps = np.sinc(2*.15*x)*np.hamming(63)
        self.taps /= self.taps.sum()
        self.tail = np.zeros(62)

    def convert(self, stereo):
        import numpy as np
        mono = np.frombuffer(stereo, dtype='<i2').reshape(-1, self.channels)[:, 0]
        if self.rate == 16000:
            return mono.copy().tobytes()
        joined = np.concatenate((self.tail, mono))
        self.tail = joined[-62:].copy()
        output = np.convolve(joined, self.taps, mode='valid')[::3]
        return np.clip(np.rint(output), -32768, 32767).astype('<i2').tobytes()


class Utterance:
    """16 kHz mono PCM, with preroll, minimum voiced time and bounded storage."""
    def __init__(self, level=200, silence=.8, maximum=20):
        self.level, self.silence, self.maximum = level, silence, maximum
        self.reset()

    def reset(self):
        self.preroll = deque(maxlen=10)  # 200 ms at 20 ms per block
        self.parts = []
        self.voiced = self.duration = self.quiet = 0

    def feed(self, pcm, rms, held=False):
        duration = len(pcm)/32000
        speech = rms >= self.level
        if not self.parts:
            self.preroll.append(pcm)
            if not speech:
                return None
            self.parts = list(self.preroll)
        else:
            self.parts.append(pcm)
        self.duration += duration
        self.voiced += duration if speech else 0
        self.quiet = 0 if speech else self.quiet+duration
        # Drop overlong/held input rather than grow without bound or split an
        # active peer's speech into competing replies.
        if self.duration >= self.maximum:
            self.reset()
            return None
        if self.quiet >= self.silence and not held:
            audio = b''.join(self.parts) if self.voiced >= .2 else None
            self.reset()
            return audio
        return None


def transcribe(url, pcm, *, response_format='json', prompt='', sound=False):
    wav = io.BytesIO()
    with wave.open(wav, 'wb') as writer:
        writer.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        writer.writeframes(pcm)
    boundary = 'voice-trio-'+uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\n{response_format}\r\n'
            + (f'--{boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n{prompt}\r\n' if prompt else '') +
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
            'Content-Type: audio/wav\r\n\r\n').encode()+wav.getvalue()+f'\r\n--{boundary}--\r\n'.encode()
    request = urllib.request.Request(url, data=body, headers={
        'Content-Type': 'multipart/form-data; boundary='+boundary,
        **({'X-Voice-Sound': '1'} if sound else {})})
    with urllib.request.urlopen(request, timeout=30) as response:
        if sound:
            if response_format != 'vtt': raise ValueError('Tagged ASR requires VTT')
            return json.load(response)
        text = response.read().decode('utf-8') if response_format == 'vtt' else json.load(response)['text'].strip()
    return text


def speech_text(text):
    """Keep raw name spelling; annotations alone are not a model turn."""
    return text if re.sub(r'\[[^]]*\]|\([^)]*\)|[\s.♪♫]+', '', text) else ''
