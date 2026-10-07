"""Figure out what a file is (format) and what a graph is (model family)."""
from pathlib import Path

ATTENTION_OPS = ("attention", "multihead", "mha")
CONV_OPS = ("conv",)
DENSE_OPS = ("linear", "gemm", "matmul")


def detect_bytes(data: bytes) -> str:
    if data.startswith(b"PK"):                 # zip container: torch>=1.6 (keras later)
        return "torch"
    if data.startswith(b"\x80"):               # legacy pickle: old torch saves
        return "torch"
    if len(data) >= 9 and data[8:9] == b"{":   # safetensors: u64 header len + JSON
        return "hf"
    if data[:1] == b"\x08":                    # protobuf field 1: onnx ir_version
        return "onnx"
    return "unknown"


def detect_format(path: str | Path) -> str:
    """Content sniffing first, extension as fallback."""
    path = Path(path)
    if path.is_dir():                          # HuggingFace model folder
        has_config = (path / "config.json").exists()
        has_weights = any(path.glob("*.safetensors"))
        return "hf" if has_config or has_weights else "unknown"
    fmt = detect_bytes(path.read_bytes()[:16])
    if fmt != "unknown":
        return fmt
    ext = path.suffix.lower()
    return {".pth": "torch", ".pt": "torch", ".onnx": "onnx",
            ".safetensors": "hf"}.get(ext, "unknown")


def classify(g) -> str:
    """Model family from layer ops: MLP / CNN / Transformer / other."""
    if any(n.kind == "head" for n in g.nodes):
        return "Transformer"
    ops = [n.op.lower() for n in g.nodes if n.kind == "layer"]
    if any(any(k in op for k in ATTENTION_OPS) for op in ops):
        return "Transformer"
    if any(any(op.startswith(k) for k in CONV_OPS) for op in ops):
        return "CNN"
    if any(any(op.startswith(k) for k in DENSE_OPS) for op in ops):
        return "MLP"
    return "other"
