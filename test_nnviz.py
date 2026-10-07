"""Self-check for nnviz. Run: python test_nnviz.py"""
from pathlib import Path

import torch.nn as nn

from nnviz.parsers import parse_module, parse_onnx
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


# --- v0.2: detection, ONNX, classification ---

def test_detect_format_by_content():
    import torch
    from nnviz.detect import detect_format
    import onnx
    from onnx import helper, TensorProto

    torch_path = Path("tmp_detect.pth")
    torch.save(nn.Sequential(nn.Linear(4, 2)).state_dict(), torch_path)
    assert detect_format(torch_path) == "torch"

    onnx_model = helper.make_model(
        helper.make_graph([], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])]))
    onnx_path = Path("tmp_detect.onnx")
    onnx.save(onnx_model, onnx_path)
    assert detect_format(onnx_path) == "onnx"

    junk_path = Path("tmp_junk.bin")
    junk_path.write_bytes(b"not a model at all")
    assert detect_format(junk_path) == "unknown"

    for p in (torch_path, onnx_path, junk_path):
        p.unlink()


def test_classify_model_type():
    from nnviz.detect import classify
    from nnviz.ir import Graph, Node

    def graph(ops):
        g = Graph()
        g.nodes = [Node(str(i), "layer", op) for i, op in enumerate(ops)]
        return g

    assert classify(graph(["Input", "Linear", "ReLU", "Linear"])) == "MLP"
    assert classify(graph(["Input", "Conv2d", "ReLU", "Linear"])) == "CNN"
    assert classify(graph(["Input", "Attention", "Linear"])) == "Transformer"
    assert classify(graph(["Input", "Gather", "Cast"])) == "other"


def test_onnx_parser_follows_real_dag_edges():
    import numpy as np
    import onnx
    from onnx import helper, TensorProto

    # Conv -> Relu -> Gemm, built as a real ONNX graph
    w_conv = helper.make_tensor("w_conv", TensorProto.FLOAT, [2, 1, 3, 3], np.zeros(18, dtype=np.float32))
    w_gemm = helper.make_tensor("w_gemm", TensorProto.FLOAT, [4, 50], np.zeros(200, dtype=np.float32))
    nodes = [
        helper.make_node("Conv", ["x", "w_conv"], ["c"], kernel_shape=[3, 3]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("Gemm", ["r", "w_gemm"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "net",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 10, 10])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [w_conv, w_gemm])
    g = parse_onnx(helper.make_model(graph))

    layers = [n for n in g.nodes if n.kind == "layer"]
    assert [n.op for n in layers] == ["Input", "Conv", "Relu", "Gemm"]
    ids = [n.id for n in layers]
    assert [(e.src, e.dst) for e in g.edges] == list(zip(ids, ids[1:]))
    conv = layers[1]
    assert conv.attrs["in_channels"] == 1
    assert conv.attrs["out_channels"] == 2
    gemm = layers[3]
    assert gemm.attrs["in_features"] == 50
    assert gemm.attrs["size"] == 4
    # pass-through ops inherit the unit count flowing through them
    assert layers[2].attrs["size"] == 2
    assert g.meta["format"] == "onnx"


# --- v0.3: make weights visible ---

def test_module_nodes_carry_weight_stats():
    import torch
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    g = parse_module(model)

    lin = next(n for n in g.nodes if n.op == "Linear" and n.attrs.get("in_features") == 4)
    ws = lin.attrs["weights"]
    assert ws["count"] == 32
    assert ws["min"] <= ws["mean"] <= ws["max"]
    assert len(ws["hist"]) == 40
    assert sum(ws["hist"]) == 32
    assert ws["hist_lo"] <= ws["min"] and ws["max"] <= ws["hist_hi"]
    # ReLU has no parameters, so it carries no weight stats
    relu = next(n for n in g.nodes if n.op == "ReLU")
    assert "weights" not in relu.attrs


def test_edge_strength_mapped_from_weights():
    import torch
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    g = parse_module(model)
    l1 = next(n for n in g.nodes if n.op == "Linear" and n.attrs.get("in_features") == 4)
    l2 = next(n for n in g.nodes if n.op == "Linear" and n.attrs.get("in_features") == 8)

    e_in = next(e for e in g.edges if e.dst == l1.id)
    assert abs(e_in.attrs["strength"] - l1.attrs["weights"]["mean_abs"]) < 1e-6
    # the link into a weightless pass-through op inherits the upstream strength
    e_relu = next(e for e in g.edges if e.dst != l1.id and e.dst != l2.id)
    assert abs(e_relu.attrs["strength"] - l1.attrs["weights"]["mean_abs"]) < 1e-6
    e_out = next(e for e in g.edges if e.dst == l2.id)
    assert abs(e_out.attrs["strength"] - l2.attrs["weights"]["mean_abs"]) < 1e-6


def test_state_dict_carries_weight_stats():
    import torch
    from nnviz.parsers import parse_state_dict
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    g = parse_state_dict(model.state_dict())
    lin = next(n for n in g.nodes if n.op == "Linear" and n.attrs.get("in_features") == 784)
    ws = lin.attrs["weights"]
    assert ws["count"] == 784 * 128
    assert len(ws["hist"]) == 40


def test_onnx_carries_weight_stats():
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    rng = np.random.default_rng(0)
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 50], rng.standard_normal(200, dtype=np.float32))
    nodes = [helper.make_node("Gemm", ["x", "w"], ["y"])]
    graph = helper.make_graph(
        nodes, "net",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 50])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], [w])
    g = parse_onnx(helper.make_model(graph))

    gemm = next(n for n in g.nodes if n.op == "Gemm")
    ws = gemm.attrs["weights"]
    assert ws["count"] == 200
    assert ws["min"] < 0 < ws["max"]
    assert abs(g.edges[0].attrs["strength"] - ws["mean_abs"]) < 1e-6


def test_serve_honors_port_flag():
    import nnviz.cli as cli
    import nnviz.server as server
    calls = {}
    orig = server.serve
    server.serve = lambda port=8000: calls.setdefault("port", port)
    try:
        cli.main(["serve", "--port", "9001"])
    finally:
        server.serve = orig
    assert calls.get("port") == 9001


def test_server_serves_generated_files():
    import threading
    from urllib.request import urlopen
    from http.server import HTTPServer
    from nnviz.server import Handler

    Path("tmp_served.html").write_text("<h1>generated</h1>", encoding="utf-8")
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        port = httpd.server_address[1]
        # the viewer lives at /
        assert "Neural Network Visualizer" in urlopen(f"http://127.0.0.1:{port}/").read().decode()
        # generated files are served from the launch directory
        assert "generated" in urlopen(f"http://127.0.0.1:{port}/tmp_served.html").read().decode()
    finally:
        httpd.shutdown()
        Path("tmp_served.html").unlink()


def test_weight_stats_include_per_neuron_strengths():
    import torch
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Conv2d(3, 6, 3))
    g = parse_module(model)
    lin = next(n for n in g.nodes if n.op == "Linear")
    rs = lin.attrs["weights"]["row_strengths"]
    assert len(rs) == 8                       # one strength per output neuron
    import numpy as np
    w = model[0].weight.detach().numpy()
    assert abs(rs[3] - float(np.abs(w[3]).mean())) < 1e-4
    conv = next(n for n in g.nodes if n.op == "Conv2d")
    assert len(conv.attrs["weights"]["row_strengths"]) == 6   # one per filter


def test_server_keeps_serving_during_slow_parse():
    import threading
    import time
    from urllib.request import urlopen
    import nnviz.server as srv

    class Slow(srv.Handler):
        def do_POST(self):
            time.sleep(3)

    # pick a free port
    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    real_open, real_handler = srv.webbrowser.open, srv.Handler
    srv.webbrowser.open = lambda *a, **k: None
    srv.Handler = Slow
    try:
        threading.Thread(target=srv.serve, args=(port,), daemon=True).start()
        time.sleep(0.7)
        threading.Thread(
            target=lambda: urlopen(f"http://127.0.0.1:{port}/parse", data=b"x", timeout=10).read(),
            daemon=True).start()
        time.sleep(0.3)                     # the slow parse now holds the server
        html = urlopen(f"http://127.0.0.1:{port}/", timeout=2).read().decode()
        assert "Neural Network Visualizer" in html
    finally:
        srv.webbrowser.open, srv.Handler = real_open, real_handler


def test_serve_ctrl_c_stops_quietly():
    import nnviz.server as srv

    class FakeServer:
        def __init__(self, addr, handler):
            pass
        def serve_forever(self):
            raise KeyboardInterrupt

    orig = srv.ThreadingHTTPServer
    srv.ThreadingHTTPServer = FakeServer
    try:
        srv.serve(0)          # must swallow Ctrl+C, no traceback
    finally:
        srv.ThreadingHTTPServer = orig


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
