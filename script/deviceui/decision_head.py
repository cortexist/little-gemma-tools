"""Resident C/GGUF decision head. No NumPy, CUDA, or subprocess dependency.

Python owns transport and scheduling; CDLL releases the GIL during C inference.
One native workspace is reused, protected against concurrent predict/close.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import sys
import threading


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8*1024*1024): digest.update(block)
    return digest.hexdigest()


class Features(ct.Structure):
    _fields_ = [('input_dim', ct.c_size_t), ('model_sha256', ct.c_char_p),
                ('input_format', ct.c_char_p), ('capture_boundary', ct.c_char_p),
                ('layer', ct.c_int32), ('answer_channel_seed', ct.c_int)]


class Spec(ct.Structure):
    _fields_ = [('architecture', ct.c_int), ('features', Features),
                ('hidden_dim', ct.c_size_t), ('output_dim', ct.c_size_t),
                ('ensemble_size', ct.c_size_t), ('clip', ct.c_float),
                ('temperature', ct.c_double), ('mean', ct.c_void_p),
                ('scale', ct.c_void_p), ('members', ct.c_void_p),
                ('labels', ct.POINTER(ct.c_char_p)), ('task', ct.c_char_p),
                ('source_contract_json', ct.c_char_p)]


class CHead:
    def __init__(self, path, library, expected_sha256=None):
        self._lock = threading.Lock()
        self._head = self._workspace = None
        if sys.byteorder != 'little':
            raise ValueError('Decision snapshots require a little-endian host')
        path, library = Path(path).expanduser().resolve(), Path(library).expanduser().resolve()
        digest = file_sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('Decision GGUF hash mismatch')
        self.lib = ct.CDLL(str(library))
        def bind(name, result, *args):
            fn = getattr(self.lib, 'decision_head_'+name)
            fn.restype, fn.argtypes = result, list(args)
            return fn
        if bind('abi_version', ct.c_uint)() != 1:
            raise ValueError('Unsupported decision-head C ABI')
        self._load = bind('load_gguf', ct.c_void_p, ct.c_char_p, ct.c_char_p, ct.c_size_t)
        self._free = bind('free', None, ct.c_void_p)
        self._spec = bind('get_spec', ct.POINTER(Spec), ct.c_void_p)
        self._check = bind('check_features', ct.c_int, ct.c_void_p, ct.POINTER(Features), ct.c_char_p, ct.c_size_t)
        self._create_workspace = bind('workspace_create', ct.c_void_p, ct.c_void_p)
        self._free_workspace = bind('workspace_free', None, ct.c_void_p)
        fp = ct.POINTER(ct.c_float)
        self._predict = bind('predict', ct.c_int, ct.c_void_p, ct.c_void_p, fp,
                             ct.c_size_t, fp, fp, ct.c_size_t, ct.POINTER(ct.c_size_t), ct.c_char_p, ct.c_size_t)
        self._error = ct.create_string_buffer(512)
        try:
            self._head = self._load(os.fsencode(path), self._error, len(self._error))
            if not self._head: raise ValueError(self._error.value.decode('utf8', 'replace'))
            spec = self._spec(self._head).contents
            self.width = spec.features.input_dim
            self.classes = [spec.labels[i].decode('utf8') for i in range(spec.output_dim)]
            self.contract = json.loads(spec.source_contract_json.decode('utf8'))
            # Embedded source provenance must agree with the actual GGUF feature
            # contract. Never apply predictions using contradictory metadata.
            self.features = dict(hidden_dim=self.width, layer=spec.features.layer,
                model_sha256=spec.features.model_sha256.decode(),
                input_format=spec.features.input_format.decode(),
                capture_boundary=spec.features.capture_boundary.decode(),
                answer_channel_seed=bool(spec.features.answer_channel_seed), task=spec.task.decode())
            for key, value in self.features.items():
                if self.contract.get(key) != value:
                    raise ValueError('Decision GGUF/source contract mismatch: '+key)
            self.info = dict(backend='c', device='cpu', format='gguf', model=str(path),
                model_sha256=digest, library=str(library), library_sha256=file_sha256(library),
                architecture='linear' if spec.architecture == 0 else 'mlp',
                hidden_dim=spec.hidden_dim, ensemble_size=spec.ensemble_size, labels=self.classes)
            self._workspace = self._create_workspace(self._head)
            if not self._workspace: raise MemoryError('Cannot allocate decision-head workspace')
            self._input = (ct.c_float*self.width)()
            self._logits = (ct.c_float*len(self.classes))()
            self._probabilities = (ct.c_float*len(self.classes))()
            self._choice = ct.c_size_t()
        except BaseException:
            self.close()
            raise

    def check_features(self, *, model_sha256, input_format, capture_boundary,
                       input_dim, layer, answer_channel_seed):
        features = Features(input_dim, model_sha256.encode(), input_format.encode(),
                            capture_boundary.encode(), layer, answer_channel_seed)
        with self._lock:
            if not self._head: raise ValueError('Decision head is closed')
            if self._check(self._head, ct.byref(features), self._error, len(self._error)):
                raise ValueError(self._error.value.decode('utf8', 'replace'))

    def predict(self, raw):
        if not isinstance(raw, bytes) or len(raw) != self.width*4:
            raise ValueError('Invalid hidden vector')
        with self._lock:
            if not self._head: raise ValueError('Decision head is closed')
            ct.memmove(self._input, raw, len(raw))
            if self._predict(self._head, self._workspace, self._input, self.width,
                             self._logits, self._probabilities, len(self.classes),
                             ct.byref(self._choice), self._error, len(self._error)):
                raise ValueError(self._error.value.decode('utf8', 'replace'))
            return dict(action=self.classes[self._choice.value], logits=list(self._logits),
                        probabilities=dict(zip(self.classes, self._probabilities)))

    def close(self):
        with self._lock:
            if self._workspace:
                self._free_workspace(self._workspace)
                self._workspace = None
            if self._head:
                self._free(self._head)
                self._head = None
