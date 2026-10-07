"""CLI: nnviz model.pth -o model.html"""
import argparse
import sys
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="nnviz", description="Visualize a neural network in 3D.")
    ap.add_argument("model", help="model file: .pth (nn.Module or state_dict)")
    ap.add_argument("-o", "--out", default=None, help="output HTML (default: <model>.html)")
    ap.add_argument("--json", default=None, help="also dump the raw IR graph as JSON")
    args = ap.parse_args(argv)

    import torch
    from .layout import layout
    from .parsers import parse_module, parse_state_dict
    from .render import render_html

    path = Path(args.model)
    try:
        # safe mode: tensors only, no code execution
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # full nn.Module pickles need code execution; say so loudly
        print(f"WARNING: {path} needs unsafe unpickling. "
              "Only load files you trust: this runs embedded code.", file=sys.stderr)
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        g = parse_module(obj)
    elif isinstance(obj, dict):
        g = parse_state_dict(obj)
    else:
        sys.exit(f"Unsupported content in {path}: {type(obj).__name__}")
    layout(g)

    out = Path(args.out) if args.out else path.with_suffix(".html")
    out.write_text(render_html(g), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(__import__("json").dumps(g.to_dict(), indent=2), encoding="utf-8")
    print(f"{out}  ({len(g.nodes)} layers, {g.meta.get('params', '?')} params)")
