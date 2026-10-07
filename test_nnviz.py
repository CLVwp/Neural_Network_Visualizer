"""Self-check for nnviz v0.1: parser, IR, layout. Run: python test_nnviz.py"""
import torch.nn as nn

from nnviz.parsers import parse_module
from nnviz.layout import layout


def test_sequential_mlp_produces_chain_graph():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    g = parse_module(model)

    layers = [n for n in g.nodes if n.kind == "layer"]  # Input + 3 modules
    assert [n.op for n in layers] == ["Input", "Linear", "ReLU", "Linear"]

    edges = [(e.src, e.dst) for e in g.edges]
    ids = [n.id for n in layers]
    assert edges == list(zip(ids, ids[1:]))

    lin = next(n for n in layers if n.op == "Linear" and n.attrs.get("out_features") == 8)
    assert lin.attrs["in_features"] == 4


def test_cnn_graph_has_conv_attrs():
    model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Flatten(), nn.Linear(16 * 222 * 222, 10))
    g = parse_module(model)
    conv = next(n for n in g.nodes if n.op == "Conv2d")
    assert conv.attrs["in_channels"] == 3
    assert conv.attrs["out_channels"] == 16


def test_layout_positions_layers_on_distinct_x():
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    g = parse_module(model)
    layout(g)
    layers = [n for n in g.nodes if n.kind == "layer"]  # Input + 3 modules
    assert all(len(n.pos) == 3 for n in layers)
    xs = [round(n.pos[0], 6) for n in layers]
    assert len(set(xs)) == len(layers)
    # neuron count is exposed for the viewer to draw dots
    assert layers[0].attrs["size"] == 784
    assert layers[3].attrs["size"] == 10
    # input layer exists so the graph starts where data enters
    assert layers[0].op == "Input"
    assert layers[3].op == "Linear"


def test_graph_serializes_to_json_dict():
    model = nn.Sequential(nn.Linear(4, 2))
    g = parse_module(model)
    layout(g)
    d = g.to_dict()
    assert set(d) == {"meta", "nodes", "edges"}
    assert all(set(n) >= {"id", "kind", "op", "pos", "attrs"} for n in d["nodes"])


def test_state_dict_infers_linear_chain():
    from nnviz.parsers import parse_state_dict
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    g = parse_state_dict(model.state_dict())

    layers = [n for n in g.nodes if n.kind == "layer"]
    # ReLU has no parameters, so a state_dict only reveals the Linear layers
    assert [n.op for n in layers] == ["Input", "Linear", "Linear"]
    assert layers[1].attrs["in_features"] == 784
    assert layers[2].attrs["size"] == 10
    ids = [n.id for n in layers]
    assert [(e.src, e.dst) for e in g.edges] == list(zip(ids, ids[1:]))


def test_render_embeds_json_and_drops_placeholder():
    import json
    from nnviz.render import render_html
    model = nn.Sequential(nn.Linear(4, 2))
    g = parse_module(model)
    layout(g)
    html = render_html(g)
    assert "__MODEL_DATA__" not in html
    assert json.dumps(g.to_dict())[:20] in html or '"nodes"' in html


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
