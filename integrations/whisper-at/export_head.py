"""Export a supplied Whisper-AT checkpoint for the classroom's 10-second head."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['base.en', 'small.en-low', 'small.en'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True,
                        help='Representative pooled encoder features, .npy [layers,75,width]')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    import numpy as np
    import onnxruntime as ort
    import torch
    from whisper_at.model import ATModel
    small = args.model.startswith('small')
    shape = (12, 75, 768) if small else (6, 75, 512)
    x = np.load(args.features, allow_pickle=False).astype(np.float32)
    if x.shape != shape or not np.isfinite(x).all():
        parser.error(f'Features must be finite with shape {shape}')
    head = ATModel(n_layer=shape[0], rep_dim=shape[2],
                   mode='tl_down_tr_512_1_8' if args.model.endswith('-low') else 'tl_tr_1_8').eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    head.load_state_dict({k.removeprefix('at_model.'): v for k, v in state.items()}, strict=True)
    torch.set_num_threads(4)

    class Window(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head

        def forward(self, features):
            return self.head(features, time_resolution=10.)

    wrapped = Window().eval()
    dummy = torch.from_numpy(x)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(wrapped, dummy, str(args.output), input_names=['encoder_layers'],
                          output_names=['logits'], opset_version=17, dynamo=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(args.output), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        with torch.inference_mode():
            expected = wrapped(dummy).numpy()
        actual = session.run(None, {'encoder_layers': x})[0]
        if actual.shape != (3, 527) or not np.isfinite(actual).all():
            raise ValueError('Invalid exported logits')
        error = float(np.max(np.abs(expected-actual)))
        if error >= .001:
            raise ValueError(f'ONNX/reference mismatch: {error}')
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(json.dumps(dict(model=args.model, feature_shape=shape,
                          max_absolute_logit_error=error,
                          sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
