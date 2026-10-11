# Whisper-AT adapter dependencies

The classroom sound adapter consumes intermediate encoder features from a
patched [whisper.cpp](https://github.com/ggml-org/whisper.cpp) server and runs a
separate CPU ONNX [Whisper-AT](https://github.com/YuanGongND/whisper-at) head.
This directory publishes the integration patch and export utility. Model weights
and upstream projects are separate dependencies with their own terms.

The supplied patch targets whisper.cpp revision
`f049fff95a089aa9969deb009cdd4892b3e74916` (v1.9.1), the Orin baseline. Apply it in
a separate checkout, then build both server and libraries together:

```sh
git checkout f049fff95a089aa9969deb009cdd4892b3e74916
git apply /path/to/little-gemma-tools/integrations/whisper-at/whisper-cpp-v1.9.1-encoder-tap.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
cmake --build build --target whisper-server -j4
```

The opt-in `WHISPER_AT_FEATURE_PATH` writes a request's encoder block outputs,
average-pooled over 20 frames before final normalization. The binary format is
`WAFT`, four little-endian uint32 values (layers, frames, width, mel offset), then
float32 values in `[layer,time,width]` order. The adapter admits only offset zero
and inputs up to 30 seconds. It serializes requests, deletes stale features and
binds the native server to its own lifetime. Do not share this native server
with unrelated clients. The patch preserves ordinary ASR when the variable is
unset. This is an internal feature-file interface, not an upstream Whisper API.

Use the Whisper-AT source at `17d94d6acd53866390ce70f95afa13507dcb8aef` and
supply its matching tagging checkpoint: `base.en_ori.pth` for base.en or
`small.en_low.pth` for small.en-low. The original project supplies acquisition
instructions; weights are not redistributed here. Install its dependencies plus
PyTorch, NumPy, ONNX and ONNX Runtime in an export environment.

Capture a representative request with the patched server, retaining its WAFT
file as `/private/encoder.bin`. Convert the pooled tensor to NumPy:

```python
from pathlib import Path
import struct
import numpy as np
blob = Path('/private/encoder.bin').read_bytes()
assert blob[:4] == b'WAFT'
layers, frames, width, offset = struct.unpack('<4I', blob[4:20])
assert offset == 0
np.save('/private/features.npy', np.frombuffer(blob[20:], dtype='<f4').reshape(layers, frames, width))
```

Export and check the ten-second pooling graph against PyTorch on that input:

```sh
python export_head.py --model base.en --checkpoint /models/base.en_ori.pth \
  --features /private/features.npy --output /models/base.en-10s.onnx
```

The emitted SHA-256 and feature shape go into `acoustic_input.head_sha256` and
`feature_shape`; set `head` to the ONNX path. E4B does not determine the Whisper
size: use `[6,75,512]` for base.en, `[12,75,768]` for small.en. Pin the built native
library hash separately. Keep the server and its matching libraries together;
do not mix whisper.cpp/ggml build versions. The export check is numerical parity
on supplied features, not validation of real-world sound recognition accuracy.

See [the acoustic contract](../../docs/acoustic-input.md) and
`script/deviceui/whisper_sound_server.py` for the private loopback server and
node configuration fields.
