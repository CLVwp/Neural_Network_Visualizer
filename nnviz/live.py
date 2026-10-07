"""Live mode: run one forward pass, capture per-layer activations."""
import io

import numpy as np
import torch
from torch import nn


def find_input_shape(model: nn.Module) -> tuple:
    """Probe the input shape: first Linear in_features, else a spatial size
    the whole model accepts."""
    first_lin = next((m for m in model.modules() if isinstance(m, nn.Linear)), None)
    first_conv = next((m for m in model.modules() if isinstance(m, nn.Conv2d)), None)
    if first_conv is None:
        if first_lin is None:
            raise ValueError("model has no Linear or Conv2d layer: cannot guess an input")
        return (1, first_lin.in_features)
    for h in (28, 32, 64, 128, 224, 96, 16, 8):  # ponytail: probe common sizes; explicit input UI if this misses
        shape = (1, first_conv.in_channels, h, h)
        try:
            with torch.no_grad():
                model(torch.zeros(shape))
            return shape
        except Exception:
            continue
    raise ValueError("could not find a working input size. Feed an image of the trained size.")


def random_batch(shape) -> torch.Tensor:
    return torch.randn(*shape)


def image_batch(shape, data: bytes) -> torch.Tensor:
    """Decode image bytes to a normalized 1CHW batch at the model's size."""
    try:
        from PIL import Image
    except ImportError:
        raise ValueError("image input needs pillow: pip install pillow")
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((shape[-1], shape[-2]))
    arr = np.asarray(img, dtype=np.float32) / 255.0            # HWC
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1CHW


def _reduce(t: torch.Tensor) -> list:
    """(1, U, ...) -> one value per unit: mean over the trailing dimensions."""
    a = t.detach().cpu().numpy().astype(np.float64)
    a = a.reshape(a.shape[0], a.shape[1], -1).mean(axis=(0, 2))
    return [round(float(v), 5) for v in a]


MAX_ACT = 4096

def run_forward(model: nn.Module, x: torch.Tensor) -> dict:
    """One forward pass. Return activations per layer + output logits/probs."""
    acts = [("in_0", _reduce(x))]
    hooks = []

    def hook(name):
        def f(_m, _i, out):
            acts.append((name, _reduce(out)))
        return f

    # leaf modules only: same set of ids the torch parser used for the graph
    for name, m in model.named_modules():
        if name and len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook(name)))
    with torch.no_grad():
        out = model(x)
    for h in hooks:
        h.remove()

    logits = out.detach().cpu().numpy().astype(np.float64).ravel()
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    def wire(a):        # ponytail: cap big vectors (Flatten) for the wire; LOD in v0.8
        return a if len(a) <= MAX_ACT else a[::int(np.ceil(len(a) / MAX_ACT))]
    return {"acts": [{"id": i, "act": wire(a)} for i, a in acts],
            "output": {"logits": [round(float(v), 5) for v in logits],
                       "probs": [float(v) for v in probs],   # unrounded: they must sum to 1
                       "pred": int(probs.argmax())}}
