"""Recording-owned, lossless ASR input WAVs and timing metadata on the hub."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request
import wave

from recording_input import BLOCK, FORMAT, FRAME, RATE


def read_exact(source, size):
    data = bytearray()
    while len(data) < size:
        part = source.read(size-len(data))
        if not part: raise RuntimeError('ASR input connection closed without a complete end record')
        data.extend(part)
    return bytes(data)


def read_frame(source):
    kind, size = FRAME.unpack(read_exact(source, FRAME.size))
    if size > 65536: raise ValueError('Oversized ASR input frame')
    return kind, read_exact(source, size)


class InputTrack:
    def __init__(self, node, session, directory, cancel):
        self.node, self.session, self.directory, self.cancel = node, session, directory, cancel
        self.ready = threading.Event()
        self.capture_id = None
        self.error = ''
        self.source = None
        self.result = dict(node=node['id'], speaker_id=node['speaker_id'], complete=False,
                           wav=node['id']+'.wav', timing=node['id']+'.jsonl')
        self.thread = threading.Thread(target=self.work, name='asr-record-'+node['id'], daemon=True)

    def request(self, suffix='', body=None):
        return urllib.request.Request(self.node['url'].rstrip('/')+'/api/recording/asr'+suffix,
            data=None if body is None else json.dumps(body).encode(),
            headers={'X-Voice-Key': self.node['key'], 'Content-Type': 'application/json'})

    def work(self):
        samples = 0
        digest = hashlib.sha256()
        try:
            self.result['hub_connect_unix'] = time.time()
            with urllib.request.urlopen(self.request('?session='+urllib.parse.quote(self.session)), timeout=5) as source:
                self.source = source
                kind, payload = read_frame(source)
                if kind != b'M': raise ValueError('Missing ASR input metadata')
                meta = json.loads(payload)
                if (meta.get('protocol') != FORMAT or meta.get('session') != self.session or
                        meta.get('speaker_id') != self.node['speaker_id'] or meta.get('rate') != RATE or
                        meta.get('channels') != 1 or meta.get('sample_format') != 's16le' or
                        not isinstance(meta.get('capture_id'), str)):
                    raise ValueError('ASR input identity, session or format mismatch')
                self.capture_id = meta['capture_id']
                self.result.update(input=meta, hub_headers_unix=time.time())
                # Exclusive creation prevents a retry from overwriting an earlier diagnostic.
                with (self.directory/self.result['wav']).open('xb') as output, \
                        (self.directory/self.result['timing']).open('x') as timing, \
                        wave.open(output, 'wb') as wav:
                    wav.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
                    while True:
                        kind, payload = read_frame(source)
                        if kind == b'E':
                            end = json.loads(payload)
                            self.result['end'] = end
                            if end.get('samples') != samples: raise ValueError('ASR input sample count mismatch')
                            if end.get('error'): raise RuntimeError(end['error'])
                            if not self.cancel.is_set(): raise RuntimeError('ASR input ended before recording stopped')
                            self.result['complete'] = True
                            break
                        if kind != b'P' or len(payload) <= BLOCK.size:
                            raise ValueError('Invalid ASR input block')
                        offset, captured, eligible = BLOCK.unpack(payload[:BLOCK.size])
                        pcm = payload[BLOCK.size:]
                        if offset != samples or len(pcm) % 2 or eligible not in (0, 1) or not math.isfinite(captured):
                            raise ValueError('Discontinuous ASR input samples')
                        wav.writeframesraw(pcm)
                        digest.update(pcm)
                        timing.write(json.dumps(dict(sample=offset, frames=len(pcm)//2,
                            captured_monotonic=captured, asr_eligible=bool(eligible)))+'\n')
                        samples += len(pcm)//2
                        if not self.ready.is_set(): self.result['hub_first_block_unix'] = time.time()
                        self.ready.set()
        except Exception as exc:
            self.error = self.node['id']+': '+str(exc)
        finally:
            self.source = None
            self.result.update(samples=samples, seconds=samples/RATE, pcm_sha256=digest.hexdigest(),
                               error=self.error, hub_closed_unix=time.time())
            self.ready.set()

    def request_stop(self):
        if not self.capture_id or not self.thread.is_alive(): return
        try:
            with urllib.request.urlopen(self.request('/stop',
                    dict(session=self.session, capture_id=self.capture_id)), timeout=3) as response:
                response.read()
        except Exception as exc:
            # A concurrently stopped node can have already emitted its terminal frame.
            # Only that verified terminal frame may establish completeness.
            self.result['stop_request_error'] = str(exc)

    def check(self):
        if self.error: raise RuntimeError(self.error)
        if self.ready.is_set() and not self.thread.is_alive() and not self.cancel.is_set():
            raise RuntimeError(self.node['id']+': ASR input stopped unexpectedly')


class InputTracks:
    def __init__(self, nodes, session):
        self.nodes, self.session = list(nodes), session
        self.tracks = []
        self.directory = None
        self.manifest = None
        self.cancel = None

    def save_manifest(self):
        if self.directory:
            data = dict(self.manifest, tracks=[dict(track.result) for track in self.tracks])
            temp = self.directory/'manifest.tmp'
            temp.write_text(json.dumps(data, indent=2)+'\n')
            temp.replace(self.directory/'manifest.json')

    def start(self, cancel, video):
        if not self.session: raise ValueError('ASR tracks require an active room session')
        self.cancel = cancel
        self.directory = Path(video).with_suffix('.asr')
        self.directory.mkdir(exist_ok=False)
        self.manifest = dict(schema=1, session=self.session, video=Path(video).name,
            rate=RATE, channels=1, sample_format='s16le',
            description='Exact continuous processed input before VAD/windowing; eligibility in per-block JSONL.',
            timing='Separate device clocks; not sample-synchronized. Node monotonic/unix anchors and hub times are retained.',
            hub_started_unix=time.time())
        self.tracks = [InputTrack(node, self.session, self.directory, cancel) for node in self.nodes]
        self.save_manifest()
        for track in self.tracks: track.thread.start()
        deadline = time.monotonic()+10
        while not cancel.wait(.02):
            self.check()
            if all(track.ready.is_set() for track in self.tracks):
                self.save_manifest()
                return
            if time.monotonic() >= deadline: raise RuntimeError('ASR input tracks did not become ready')

    def video_started(self, at):
        self.manifest['hub_video_ready_unix'] = at
        self.save_manifest()

    def check(self):
        for track in self.tracks: track.check()

    def close(self):
        if not self.tracks: return
        self.cancel.set()
        with ThreadPoolExecutor(max_workers=len(self.tracks)) as pool:
            list(pool.map(lambda track: track.request_stop(), self.tracks))
        deadline = time.monotonic()+7
        for track in self.tracks:
            if track.thread.ident is not None:
                track.thread.join(timeout=max(0, deadline-time.monotonic()))
            if track.thread.is_alive():
                track.error = track.node['id']+': ASR track did not finalize'
                track.result.update(complete=False, error=track.error)
        self.manifest['hub_closed_unix'] = time.time()
        self.save_manifest()
        errors = [track.error for track in self.tracks if track.error]
        if errors: raise RuntimeError('; '.join(errors))
