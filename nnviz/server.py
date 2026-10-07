"""Drag & drop server: serves the viewer, parses uploaded model files.

POST /parse with a file body -> JSON {"graph": {...}} or {"error": "..."}.
POST /run -> live mode: one forward pass on the retained module,
JSON {"acts": [...], "output": {...}}.
"""
import io
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer

from .detect import classify, detect_bytes
from .layout import layout
from .render import TEMPLATE, render_html

MAX_UPLOAD = 2 * 1024 ** 3  # 2 GiB: big LLM checkpoints are in scope
_STATE = {"model": None, "graph": None}


def live_load(graph, model) -> None:
    """Load a module + graph for live runs (nnviz live model.pth)."""
    _STATE["model"] = model
    _STATE["graph"] = graph


def _parse_bytes(data: bytes):
    """Returns (graph, module_or_None)."""
    import torch
    from .parsers import parse_module, parse_onnx, parse_state_dict

    fmt = detect_bytes(data[:16])
    if fmt == "onnx":
        return parse_onnx(onnx_load(io.BytesIO(data))), None
    if fmt == "torch":
        try:
            obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        except Exception:
            print(f"WARNING: uploaded torch file needs unsafe unpickling "
                  f"({len(data)/1e6:.0f} MB). Only drop files you trust.", file=sys.stderr)
            obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if isinstance(obj, torch.nn.Module):
            return parse_module(obj), obj
        if isinstance(obj, dict):
            g = parse_state_dict(obj)
            g.meta["params"] = sum(t.numel() for t in obj.values() if hasattr(t, "numel"))
            return g, None
        raise ValueError(f"unsupported torch content: {type(obj).__name__}")
    raise ValueError(f"unknown or unsupported format (detected: {fmt}). "
                     "Supported: PyTorch (.pth/.pt), ONNX (.onnx).")


def onnx_load(buf):
    import onnx
    return onnx.load(buf)


class Handler(SimpleHTTPRequestHandler):
    """Viewer at /, static files (generated HTML) from the launch directory."""
    timeout = 60                          # drop dead uploads instead of waiting forever

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/":
            super().do_GET()            # generated model.html files
            return
        if _STATE["graph"] is not None:  # live mode: page opens with the loaded graph
            data = json.dumps(_STATE["graph"].to_dict())
        else:
            data = json.dumps({"meta": {"format": "none"}, "nodes": [], "edges": []})
        html = TEMPLATE.read_text(encoding="utf-8")
        self._send(200, html.replace("__MODEL_DATA__", data).encode(), "text/html")

    def do_POST(self) -> None:
        if self.path == "/run":
            self._run_live()
            return
        if self.path != "/parse":
            self._send(404, b'{"error": "not found"}', "application/json")
            return
        size = int(self.headers.get("Content-Length", 0))
        if size > MAX_UPLOAD:
            self._send(413, b'{"error": "file too large"}', "application/json")
            return
        try:
            g, model = _parse_bytes(self.rfile.read(size))
            _STATE["model"] = model          # retained for live runs
            _STATE["graph"] = None           # the client renders what it just received
            g.meta["model_type"] = classify(g)
            layout(g)
            body = json.dumps({"graph": g.to_dict()}).encode()
            self._send(200, body, "application/json")
        except Exception as e:
            msg = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self._send(400, msg, "application/json")

    def _run_live(self) -> None:
        from .live import find_input_shape, image_batch, random_batch, run_forward

        model = _STATE["model"]
        if model is None:
            self._send(400, json.dumps({"error": "no runnable model. Live mode needs a "
                                       "full PyTorch nn.Module (a state_dict cannot run)."
                                       }).encode(), "application/json")
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            shape = find_input_shape(model)
            if self.headers.get("Content-Type", "").startswith("image/"):
                x = image_batch(shape, body)
            else:
                x = random_batch(shape)
            self._send(200, json.dumps(run_forward(model, x)).encode(), "application/json")
        except Exception as e:
            msg = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self._send(400, msg, "application/json")

    def log_message(self, fmt, *args) -> None:  # quiet
        pass


def serve(port: int = 8000) -> None:
    url = f"http://localhost:{port}"
    print(f"Serving on {url}  (Ctrl+C to stop). Drop a model file on the page.")
    webbrowser.open(url)
    # one thread per request: a slow parse never blocks other drops
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
