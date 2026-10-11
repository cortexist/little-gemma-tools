#!/usr/bin/env python3
"""Replay completed clean utterances through CAM++ and a local Gemma tagger.

This is an offline integration harness, not a microphone/overlap detector.
Input JSONL: {turn_id,wav,transcript,clean}; {reset:true} starts a new session.
Config JSON: {model,model_sha256,known:[{name,wav,trusted_introducer}]}.
WAV paths are relative to their respective JSON files. Use mono PCM16, 16 kHz.
Only explicitly supplied acquaintances are enrolled; there is no room roster.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from session_speakers import SpeakerSession, unit


class AcousticEncoder:
    def __init__(self, model, sha256):
        if hashlib.sha256(Path(model).read_bytes()).hexdigest() != sha256:
            raise ValueError('Speaker embedding model hash mismatch')
        for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[key] = '1'
        import numpy as np
        import sherpa_onnx
        self.np = np
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=1, provider='cpu'))

    def wav(self, path):
        with wave.open(str(path), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError('Expected mono PCM16 WAV at 16 kHz')
            duration = wav.getnframes()/16000
            if not .75 <= duration <= 30:
                raise ValueError('Expected a completed utterance of 0.75–30 seconds')
            pcm = wav.readframes(wav.getnframes())
        return self.pcm(pcm), duration

    def pcm(self, pcm, *, minimum=.75):
        if len(pcm) % 2 or not minimum <= len(pcm)/32000 <= 30:
            raise ValueError('Invalid duration of mono PCM16 at 16 kHz')
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=self.np.frombuffer(pcm, '<i2').astype(self.np.float32)/32768)
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            raise ValueError('Utterance has insufficient embedding evidence')
        return unit(self.extractor.compute(stream))


def model_proposal(path, prompt, timeout=120):
    # Each connection starts a fresh model context. The explicit, listener-local
    # observation history owns memory and reset, not an old hidden model session.
    encoded = ' '.join(prompt.split()).encode('utf-8')
    if len(encoded) > 7500:
        raise ValueError('Identity prompt exceeds the engine line budget')
    start = time.monotonic()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(encoded+b'\n')
        response = b''
        while b'<turn|>' not in response:
            part = client.recv(4096)
            if not part:
                raise RuntimeError('Model disconnected before completing its response')
            response += part
            if len(response) > 32000 or time.monotonic()-start > timeout:
                raise TimeoutError('Identity response limit exceeded')
    return response.split(b'<turn|>', 1)[0].decode('utf-8'), time.monotonic()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model-socket', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    encoder = AcousticEncoder(args.config.parent/config['model'], config['model_sha256'])
    known = []
    for prior in config.get('known', []):
        embedding, _ = encoder.wav(args.config.parent/prior['wav'])
        known.append(dict(name=prior['name'], embedding=embedding,
                          trusted_introducer=prior.get('trusted_introducer', False)))
    session = SpeakerSession(known)
    with args.output.open('x', encoding='utf-8') as output:
        for line in args.input.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            if item == {'reset': True}:
                result = dict(action='reset', state=session.reset())
            else:
                embedding, duration = encoder.wav(args.input.parent/item['wav'])
                observed = session.observe(item['turn_id'], embedding, item['transcript'],
                                           clean=item.get('clean', False), duration=duration)
                raw, elapsed = model_proposal(args.model_socket, session.prompt())
                result = dict(observed=observed, model_output=raw, model_seconds=elapsed,
                              learning=session.accept(raw, metadata_only=True), state=session.snapshot())
            output.write(json.dumps(result, ensure_ascii=False)+'\n')
            output.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
