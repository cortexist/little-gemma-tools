"""Asynchronous decision readout of immutable engine hidden-state snapshots.

The caller chooses shadow observation or a same-turn control callback. Live
conversation scores are experimental and have no calibration guarantee.
"""
import hashlib
import json
import math
from pathlib import Path
import queue
import struct
import threading
import time


MAGIC = b'\x01HLGH1'
HEADER = struct.Struct('<6I2d')
MAX_DIM = 65536


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8*1024*1024): h.update(block)
    return h.hexdigest()


class HiddenReplyDecoder:
    """Split framed snapshots from raw text across arbitrary socket boundaries."""
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.buffer = bytearray()
        self.turn = 0

    def feed(self, data):
        self.buffer.extend(data)
        text = []
        while self.buffer:
            at = self.buffer.find(b'\x01')
            if at < 0:
                text.append(bytes(self.buffer)); self.buffer.clear(); break
            if at:
                text.append(bytes(self.buffer[:at])); del self.buffer[:at]
            if len(self.buffer) < len(MAGIC):
                if not MAGIC.startswith(self.buffer): raise ValueError('Invalid hidden frame marker')
                break
            if self.buffer[:6] != MAGIC: raise ValueError('Invalid hidden frame marker')
            if len(self.buffer) < 10: break
            size, = struct.unpack_from('<I', self.buffer, 6)
            if not HEADER.size+4 <= size <= HEADER.size+4*MAX_DIM:
                raise ValueError('Invalid hidden frame size')
            if len(self.buffer) < 10+size: break
            version, turn, dim, layer, pos, seeded, at, copy_s = HEADER.unpack_from(self.buffer, 10)
            if (version != 1 or turn != self.turn+1 or not 0 < dim <= MAX_DIM
                    or size != HEADER.size+4*dim or not 0 < layer <= 1024
                    or seeded not in (0, 1) or not math.isfinite(at)
                    or not math.isfinite(copy_s) or copy_s < 0):
                raise ValueError('Invalid hidden frame metadata/sequence')
            hidden = bytes(self.buffer[10+HEADER.size:10+size])
            del self.buffer[:10+size]
            self.turn = turn
            self.snapshot(dict(turn=turn, dim=dim, layer=layer, position=pos,
                               answer_channel_seed=bool(seeded), snapshot_at=at,
                               snapshot_copy_s=copy_s, received_at=time.time(),
                               received_monotonic=time.monotonic(), hidden=hidden))
        return text

    def finish(self):
        if self.buffer: raise ValueError('Truncated hidden frame')


class NumpyHead:
    """Three-seed exported probe, single-threaded vector operations, no CUDA."""
    def __init__(self, directory):
        import numpy as np
        root = Path(directory).expanduser()
        self.contract = json.loads((root/'contract.json').read_text(encoding='utf8'))
        path = root/'head.npz'
        if sha256(path) != self.contract['head_sha256']: raise ValueError('Decision head hash mismatch')
        if self.contract.get('task') != 'final_prompt-action': raise ValueError('Expected start-turn head')
        with np.load(path, allow_pickle=False) as pack:
            self.arrays = {k: pack[k] for k in pack.files}
        a = self.arrays
        self.classes = a['classes'].tolist()
        if set(self.classes) != {'speak', 'defer', 'wait'} or len(self.classes) != 3:
            raise ValueError('Invalid decision classes')
        self.width = int(self.contract['hidden_dim'])
        if a['mean'].shape != (self.width,) or a['scale'].shape != (self.width,):
            raise ValueError('Invalid decision normalization width')
        if not np.all(a['scale'] > 0) or not float(a['temperature']) > 0:
            raise ValueError('Invalid decision normalization/temperature')
        if str(a['architecture']) not in ('linear', 'mlp64'): raise ValueError('Unsupported head architecture')
        for key, value in a.items():
            if value.dtype.kind in 'fiu' and not np.isfinite(value).all():
                raise ValueError('Nonfinite head weights: '+key)

    def predict(self, raw):
        import numpy as np
        a = self.arrays
        x = np.frombuffer(raw, dtype='<f4')
        if x.shape != (self.width,) or not np.isfinite(x).all(): raise ValueError('Invalid hidden vector')
        x = np.clip((x-a['mean'])/a['scale'], -10, 10)
        outputs = []
        # einsum deliberately avoids a BLAS thread pool contending with ASR and
        # CAM++. These are one-vector readouts, not a batch training workload.
        for i in range(3):
            p = f'model{i}.net.'
            if str(a['architecture']) == 'linear':
                z = np.einsum('ij,j->i', a[p+'weight'], x)+a[p+'bias']
            else:
                h = np.tanh(np.einsum('ij,j->i', a[p+'0.weight'], x)+a[p+'0.bias'])
                z = np.einsum('ij,j->i', a[p+'3.weight'], h)+a[p+'3.bias']
            outputs.append(z)
        logits = np.mean(outputs, axis=0)
        z = logits/float(a['temperature']); z -= z.max()
        prob = np.exp(z); prob /= prob.sum()
        return dict(action=self.classes[int(prob.argmax())], logits=logits.tolist(),
                    probabilities=dict(zip(self.classes, prob.tolist())))


class ShadowHead:
    """CPU readout; optional immediate control callback before diagnostic I/O."""
    def __init__(self, directory, model, output, result, error, save_hidden=True, decision=None,
                 backend='numpy', library=None, expected_sha256=None,
                 input_format='trio_conversational_context_v1',
                 capture_boundary='post-output-norm final prompt token; before any generated-token forward'):
        if backend == 'numpy':
            self.head = NumpyHead(directory)
            self.runtime = dict(backend='numpy', device='cpu', format='npz',
                                model=str(Path(directory)/'head.npz'))
        elif backend == 'c':
            from decision_head import CHead
            if not library or not expected_sha256:
                raise ValueError('C decision head requires a library and pinned GGUF hash')
            self.head = CHead(directory, library, expected_sha256)
            self.runtime = self.head.info
        else:
            raise ValueError('Unknown decision-head backend')
        try:
            self.model_sha256 = sha256(model)
            if self.model_sha256 != self.head.contract['model_sha256']:
                raise ValueError('Decision head backbone hash mismatch')
            if (self.head.contract.get('task') != 'final_prompt-action'
                    or set(self.head.classes) != {'speak', 'defer', 'wait'} or len(self.head.classes) != 3):
                raise ValueError('Expected a speak/defer/wait start-turn head')
            self.input_format, self.capture_boundary = input_format, capture_boundary
            if backend == 'c':
                self.head.check_features(model_sha256=self.model_sha256,
                    input_format=input_format, capture_boundary=capture_boundary,
                    input_dim=self.head.width, layer=self.head.contract['layer'],
                    answer_channel_seed=self.head.contract['answer_channel_seed'])
            self.output = Path(output)
            self.output.mkdir(parents=True, exist_ok=True)
        except BaseException:
            if hasattr(self.head, 'close'): self.head.close()
            raise
        self.result, self.error = result, error
        self.decision = decision
        self.save_hidden = save_hidden
        self.jobs = queue.Queue(maxsize=8)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.work, name='decision-head', daemon=True)
        self.thread.start()

    def submit(self, snapshot, context):
        if self.stop.is_set(): return False
        try: self.jobs.put_nowait((snapshot, context))
        except queue.Full:
            self.error('Control queue full; failing closed' if self.decision else
                       'Shadow queue full; diagnostic skipped, conversation unchanged', context)
            return False
        return True

    def work(self):
        try:
            self._work()
        finally:
            if hasattr(self.head, 'close'): self.head.close()

    def _work(self):
        while not self.stop.is_set():
            try: job = self.jobs.get(timeout=.1)
            except queue.Empty: continue
            if job is None: break
            snap, context = job
            try:
                c = self.head.contract
                if snap['dim'] != self.head.width or snap['layer'] != c['layer']:
                    raise ValueError('Snapshot backbone dimensions differ from the head')
                started = time.monotonic()
                if self.runtime['backend'] == 'c':
                    self.head.check_features(model_sha256=self.model_sha256,
                        input_format=self.input_format, capture_boundary=self.capture_boundary,
                        input_dim=snap['dim'], layer=snap['layer'],
                        answer_channel_seed=snap['answer_channel_seed'])
                prediction = self.head.predict(snap['hidden'])
                ended = time.monotonic()
                result = dict(prediction, mode='control' if self.decision else 'shadow', controlling=bool(self.decision),
                    contract_status='live_prompt_unvalidated', task='final_prompt-action',
                    runtime=self.runtime,
                    head_sha256=c['head_sha256'], model_sha256=c['model_sha256'],
                    expected_answer_channel_seed=c['answer_channel_seed'],
                    answer_channel_matches=snap['answer_channel_seed'] == c['answer_channel_seed'],
                    inference_s=ended-started, queue_s=started-snap['received_monotonic'],
                    ready_after_receive_s=ended-snap['received_monotonic'], ready_at=time.time(),
                    ready_monotonic=ended,
                    snapshot={k:v for k,v in snap.items() if k not in ('hidden', 'received_monotonic')},
                    hidden_sha256=hashlib.sha256(snap['hidden']).hexdigest())
                if self.stop.is_set(): break
                # A controlling client must receive the prediction before
                # diagnostic disk I/O. Shadow mode retains its old callback.
                if self.decision: self.decision(result, context)
                # Persist after timing inference. No conversation thread waits
                # for this disk I/O, and the snapshot owns its immutable bytes.
                if self.save_hidden:
                    name = f"c{context['connection']}-t{snap['turn']}.f32"
                    with (self.output/name).open('xb') as f: f.write(snap['hidden'])
                    result['hidden_file'] = name
                with (self.output/'decisions.jsonl').open('a', encoding='utf8') as f:
                    f.write(json.dumps(dict(context=context, **result), allow_nan=False)+'\n')
                if not self.stop.is_set(): self.result(result, context)
            except Exception as exc:
                if not self.stop.is_set(): self.error(str(exc), context)

    def close(self):
        self.stop.set()
        try: self.jobs.put_nowait(None)
        except queue.Full: pass
        self.thread.join(timeout=2)
