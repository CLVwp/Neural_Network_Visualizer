"""CLI: nnviz model.pth -o model.html | nnviz serve"""
import argparse
import sys
from pathlib import Path

BANNER = r"""
   _   _                __     __            _     _
  | \ | | _____      __  \ \   / /_ _ _   _| |__ | | ___
  |  \| |/ _ \ \ /\ / /___\ \ / / _` | | | | '_ \| |/ _ \
  | |\  |  __/\ V  V /_____\ V / (_| | |_| | |_) | |  __/
  |_| \_|\___| \_/\_/      \_/ \__,_|\__,_|_.__/|_|\___|
  visualize any neural network in 3D
"""


def _torch_load(path: Path):
    """torch.load with the loud unsafe-unpickling fallback."""
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # full nn.Module pickles need code execution; say so loudly
        print(f"WARNING: {path} needs unsafe unpickling. "
              "Only load files you trust: this runs embedded code.", file=sys.stderr)
        return torch.load(path, map_location="cpu", weights_only=False)


def _parse_any(path: Path):
    """Auto-detect the format, parse with the matching parser."""
    import torch
    from .detect import detect_format
    from .parsers import parse_module, parse_onnx, parse_state_dict

    fmt = detect_format(path)
    if fmt == "onnx":
        return parse_onnx(str(path))
    if fmt == "torch":
        obj = _torch_load(path)
        if isinstance(obj, torch.nn.Module):
            return parse_module(obj)
        if isinstance(obj, dict):
            g = parse_state_dict(obj)
            g.meta["params"] = sum(t.numel() for t in obj.values() if hasattr(t, "numel"))
            return g
        sys.exit(f"Unsupported torch content in {path}: {type(obj).__name__}")
    sys.exit(f"Unknown or unsupported format: {path}\n"
             "Supported: PyTorch (.pth, .pt), ONNX (.onnx).")


def _cmd_load(args) -> None:
    from .detect import classify
    from .layout import layout
    from .render import render_html

    path = Path(args.model)
    g = _parse_any(path)
    g.meta["model_type"] = classify(g)
    layout(g)

    out = Path(args.out) if args.out else path.with_suffix(".html")
    out.write_text(render_html(g), encoding="utf-8")
    if args.json:
        import json
        Path(args.json).write_text(json.dumps(g.to_dict(), indent=2), encoding="utf-8")
    print(f"{out}  ({g.meta['model_type']} · {len(g.nodes)} layers · "
          f"{g.meta.get('params', '?')} params)")


def _cmd_serve(args) -> None:
    from .server import serve
    serve(args.port)


def _cmd_live(args) -> None:
    """Live mode: load a full nn.Module, serve the viewer, run forward passes."""
    import torch
    from .detect import classify
    from .layout import layout
    from .parsers import parse_module
    from . import server

    path = Path(args.model)
    obj = _torch_load(path)
    if not isinstance(obj, torch.nn.Module):
        sys.exit(f"Live mode needs a full nn.Module (a {type(obj).__name__} cannot run).")
    g = parse_module(obj)
    g.meta["model_type"] = classify(g)
    layout(g)
    server.live_load(g, obj)
    print(f"Live model: {path}  ({g.meta['model_type']} · {len(g.nodes)} layers · "
          f"{g.meta.get('params', '?')} params). Use the Live panel in the page.")
    server.serve(args.port)


def main(argv=None) -> None:
    print(BANNER)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("load", "serve", "live", "-h", "--help"):
        argv.insert(0, "load")      # bare path: nnviz model.pth == nnviz load model.pth
    ap = argparse.ArgumentParser(prog="nnviz", description="Visualize a neural network in 3D.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_load = sub.add_parser("load", help="convert a model file to a 3D HTML viewer")
    p_load.add_argument("model", help="model file: .pth / .pt / .onnx")
    p_load.add_argument("-o", "--out", help="output HTML (default: <model>.html)")
    p_load.add_argument("--json", help="also dump the raw IR graph as JSON")
    p_serve = sub.add_parser("serve", help="drag & drop server: parse files in the browser")
    p_serve.add_argument("--port", type=int, default=8000)
    p_live = sub.add_parser("live", help="live mode: run the model and watch activations")
    p_live.add_argument("model", help="full PyTorch model file: .pth / .pt")
    p_live.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        _cmd_serve(args)
    elif args.cmd == "live":
        _cmd_live(args)
    else:
        _cmd_load(args)


if __name__ == "__main__":
    main()
