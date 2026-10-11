"""Native FFI equivalence and fail-closed live-worker integration.

Build sibling little-gemma's decision_head target first. Override the paths with
DECISION_HEAD_LIBRARY and DECISION_HEAD_EXPORTER for native Windows/CI builds.
"""
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest

import test_turn_head as fixtures
from test_turn_head import HiddenReplyDecoder, ShadowHead, frame
from turn_head import NumpyHead, sha256
from decision_head import CHead

ENGINE = Path(__file__).resolve().parents[2]/'little-gemma'
LIBRARY = os.environ.get('DECISION_HEAD_LIBRARY', str(ENGINE/'build'/
    ('Release/decision_head.dll' if sys.platform == 'win32' else 'libdecision_head.so')))
EXPORTER = os.environ.get('DECISION_HEAD_EXPORTER', str(ENGINE/'tools/export_decision_head.py'))
spec = importlib.util.spec_from_file_location('exporter', EXPORTER)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fixture = fixtures.ReadoutTests()

    def tearDown(self):
        self.tmp.cleanup()

    def package(self, arch='mlp64'):
        self.fixture.package(self.root, arch)
        path = self.root/'contract.json'
        contract = json.loads(path.read_text())
        contract.update(input_format='trio_conversational_context_v1',
            capture_boundary='post-output-norm final prompt token; before any generated-token forward')
        path.write_text(json.dumps(contract))
        self.gguf = self.root/(arch+'.gguf')
        exporter.write_gguf(self.gguf, *exporter.package(self.root))
        return contract

    def test_linear_and_mlp_match_numpy_and_release_resources(self):
        import numpy as np
        for arch in ('linear', 'mlp64'):
            self.package(arch)
            native = CHead(self.gguf, LIBRARY, sha256(self.gguf))
            reference = NumpyHead(self.root)
            try:
                for x in np.random.default_rng(124).normal(size=(32, 4)).astype('<f4')*30:
                    got, want = native.predict(x.tobytes()), reference.predict(x.tobytes())
                    self.assertEqual(got['action'], want['action'])
                    np.testing.assert_allclose(got['logits'], want['logits'], atol=3e-5, rtol=2e-5)
                    np.testing.assert_allclose(list(got['probabilities'].values()),
                                               list(want['probabilities'].values()), atol=1e-5)
                for raw in (bytes(4), struct.pack('<4f', float('nan'), 0, 0, 0)):
                    with self.assertRaises(ValueError): native.predict(raw)
            finally:
                native.close(); native.close()
            with self.assertRaises(ValueError): native.predict(bytes(16))

    def test_missing_native_assets_or_wrong_hash_never_fall_back(self):
        self.package()
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            CHead(self.gguf, LIBRARY, '0'*64)
        with self.assertRaises(OSError): CHead(self.gguf, self.root/'missing-library')
        with self.assertRaises(ValueError):
            ShadowHead(self.gguf, self.fixture.model, self.root/'out', None, None, backend='c')
        self.gguf.write_bytes(b'bad GGUF')
        with self.assertRaises(ValueError): CHead(self.gguf, LIBRARY)

    def test_actual_snapshot_contract_is_checked_before_control_and_disk_io(self):
        for mismatch in (None, 'layer', 'dim', 'answer_channel_seed'):
            if not hasattr(self, 'gguf'): self.package()
            output = self.root/str(mismatch)
            selected, errors = [], []
            done = threading.Event()
            def decision(result, context):
                self.assertEqual(list(output.iterdir()), [])
                selected.append(result)
            worker = ShadowHead(self.gguf, self.fixture.model, output,
                lambda *args: done.set(), lambda message, ctx: (errors.append(message), done.set()),
                decision=decision, backend='c', library=LIBRARY, expected_sha256=sha256(self.gguf))
            def submit(snap):
                if mismatch: snap[mismatch] = 1
                worker.submit(snap, dict(connection=1))
            try:
                HiddenReplyDecoder(submit).feed(frame())
                self.assertTrue(done.wait(2))
                if mismatch:
                    self.assertEqual(selected, [])
                    self.assertEqual(len(errors), 1)
                    self.assertEqual(list(output.iterdir()), [])
                else:
                    self.assertEqual(errors, [])
                    self.assertEqual(selected[0]['runtime']['backend'], 'c')
                    self.assertTrue(selected[0]['controlling'])
            finally: worker.close()
            self.assertFalse(worker.thread.is_alive())
            with self.assertRaises(ValueError): worker.head.predict(bytes(16))

    def test_host_backbone_and_prompt_contract_must_match(self):
        self.package()
        for override in (dict(input_format='structured_decision_v1'),
                         dict(capture_boundary='after decoding')):
            with self.assertRaises(ValueError):
                ShadowHead(self.gguf, self.fixture.model, self.root/'out', None, None,
                    backend='c', library=LIBRARY, expected_sha256=sha256(self.gguf), **override)
        self.fixture.model.write_bytes(b'wrong backbone')
        with self.assertRaisesRegex(ValueError, 'backbone hash'):
            ShadowHead(self.gguf, self.fixture.model, self.root/'out', None, None,
                backend='c', library=LIBRARY, expected_sha256=sha256(self.gguf))


if __name__ == '__main__': unittest.main()
