"""Binary transport, readout parity and isolation from the speech path."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'script/deviceui'),
               str(Path(__file__).resolve().parents[1]/'script')]
from turn_head import HiddenReplyDecoder, HEADER, MAGIC, NumpyHead, ShadowHead


def frame(turn=1, raw=None):
    raw = raw if raw is not None else struct.pack('<4f', 1, 2, 3, 4)
    body = HEADER.pack(1, turn, len(raw)//4, 42, 75, 0, time.time(), .00002)+raw
    return MAGIC+struct.pack('<I', len(body))+body


class TransportTests(unittest.TestCase):
    def test_every_fragment_boundary_preserves_text_and_exact_snapshot(self):
        # Marker-looking bytes inside a payload must remain opaque.
        raw = b'\x01HLGH1xx'+bytes(8)
        wire = frame(raw=raw)+b'Hello.<turn|>'+frame(2)+b'[defer]<turn|>'
        for split in range(len(wire)+1):
            snapshots = []
            decoder = HiddenReplyDecoder(snapshots.append)
            text = decoder.feed(wire[:split])+decoder.feed(wire[split:]); decoder.finish()
            self.assertEqual(b''.join(text), b'Hello.<turn|>[defer]<turn|>')
            self.assertEqual([r['turn'] for r in snapshots], [1, 2])
            self.assertEqual(snapshots[0]['hidden'], raw)

    def test_bytewise_and_invalid_frames_never_reach_text(self):
        snapshots = []; d = HiddenReplyDecoder(snapshots.append); text = []
        for b in frame()+b'[defer]<turn|>': text.extend(d.feed(bytes([b])))
        d.finish(); self.assertEqual(b''.join(text), b'[defer]<turn|>')
        for wire in (MAGIC+struct.pack('<I', 2**32-1), frame(2), b'\x01wrong',
                     MAGIC+struct.pack('<I', HEADER.size+4)+HEADER.pack(1,1,2,42,0,0,0.,0.)+bytes(4)):
            with self.assertRaises(ValueError): HiddenReplyDecoder(lambda x: None).feed(wire)
        d = HiddenReplyDecoder(lambda x: None); d.feed(frame()[:-1])
        with self.assertRaises(ValueError): d.finish()


class ReadoutTests(unittest.TestCase):
    def package(self, root, architecture):
        import numpy as np
        rng = np.random.default_rng(42)
        self.model = root/'model'; self.model.write_bytes(b'fixture backbone')
        arrays = dict(mean=np.zeros(4, np.float32), scale=np.ones(4, np.float32),
            architecture=np.array(architecture), classes=np.array(['speak','defer','wait']),
            temperature=np.array(1.5))
        for i in range(3):
            p = f'model{i}.net.'
            if architecture == 'linear':
                arrays[p+'weight'] = rng.normal(size=(3,4)).astype('f4')
                arrays[p+'bias'] = rng.normal(size=3).astype('f4')
            else:
                for name, shape in [('0.weight',(64,4)), ('0.bias',(64,)), ('3.weight',(3,64)), ('3.bias',(3,))]:
                    arrays[p+name] = rng.normal(size=shape).astype('f4')
        np.savez(root/'head.npz', **arrays)
        contract = dict(task='final_prompt-action', hidden_dim=4, layer=42, answer_channel_seed=False,
            head_sha256=hashlib.sha256((root/'head.npz').read_bytes()).hexdigest(),
            model_sha256=hashlib.sha256(self.model.read_bytes()).hexdigest())
        (root/'contract.json').write_text(json.dumps(contract))
        return arrays

    def test_single_vector_matches_exported_batch_operations(self):
        import numpy as np
        for arch in ('linear','mlp64'):
            with tempfile.TemporaryDirectory() as tmp:
                a = self.package(Path(tmp), arch); head = NumpyHead(tmp)
                for x in (np.array([1,2,3,4], 'f4'), np.array([100,-90,0,4], 'f4')):
                    z = np.clip(x, -10, 10); outputs = []
                    for i in range(3):
                        p = f'model{i}.net.'
                        outputs.append(z@a[p+'weight'].T+a[p+'bias'] if arch == 'linear' else
                            np.tanh(z@a[p+'0.weight'].T+a[p+'0.bias'])@a[p+'3.weight'].T+a[p+'3.bias'])
                    result = head.predict(x.tobytes())
                    np.testing.assert_allclose(result['logits'], np.mean(outputs, axis=0), atol=3e-5, rtol=2e-5)
                with self.assertRaises(ValueError): head.predict(bytes(4))
                with self.assertRaises(ValueError): head.predict(np.full(4, np.nan, 'f4').tobytes())
                (Path(tmp)/'head.npz').write_bytes(b'corrupted')
                with self.assertRaises(ValueError): NumpyHead(tmp)

    def test_slow_worker_submission_is_bounded_and_does_not_block_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.package(root, 'linear')
            completed = threading.Event(); entered = threading.Event(); release = threading.Event()
            errors = []; results = []
            worker = ShadowHead(root, self.model, root/'out',
                lambda result, ctx: (results.append(result), completed.set()),
                lambda error, ctx: errors.append(error))
            original = worker.head.predict
            def slow(raw):
                entered.set(); release.wait(2); return original(raw)
            worker.head.predict = slow
            decoder = HiddenReplyDecoder(lambda snap: worker.submit(snap, dict(connection=1)))
            try:
                text = decoder.feed(frame()+b'Immediate speech.<turn|>')
                self.assertEqual(b''.join(text), b'Immediate speech.<turn|>')
                self.assertTrue(entered.wait(1)); self.assertFalse(completed.is_set())
                for turn in range(2,12): decoder.feed(frame(turn))
                self.assertTrue(any('queue full' in error for error in errors))
                release.set(); self.assertTrue(completed.wait(2))
                self.assertFalse(results[0]['controlling'])
                self.assertEqual(results[0]['contract_status'], 'live_prompt_unvalidated')
            finally: release.set(); worker.close()


class IsolationTests(unittest.TestCase):
    def test_shadow_disagreement_has_no_effect_on_response_audio_or_budget(self):
        from test_room_direct import runtime
        r = runtime(); r.response_serial = 1
        r.responses['1'] = dict(id='1', heard='h1', text_action='speak', permission=True,
                                spoken=True, status='speaking')
        r.current = r.responses['1']; r.playing = True
        r.shadow_result(dict(action='defer', controlling=False), dict(session=r.session, response='1'))
        self.assertFalse(r.yielded.is_set()); self.assertTrue(r.playing)
        self.assertTrue(r.current['permission']); self.assertEqual(r.client.stdin.getvalue(), b'')
        events = r.device.emit.call_args_list
        self.assertTrue(any(c.args[0] == 'turn_head_comparison' and not c.kwargs['agrees'] for c in events))
        r.cancel.set(); r.shadow_result(dict(action='wait'), dict(session=r.session, response='1'))
        self.assertEqual(r.current['head']['action'], 'defer')


if __name__ == '__main__': unittest.main()
