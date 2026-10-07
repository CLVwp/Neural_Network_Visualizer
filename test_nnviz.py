"""Self-check for nnviz. Run: python test_nnviz.py"""
import shutil
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

    orig_server, orig_open = srv.ThreadingHTTPServer, srv.webbrowser.open
    srv.ThreadingHTTPServer = FakeServer
    srv.webbrowser.open = lambda *a, **k: None   # no browser tab from a test
    try:
        srv.serve(0)          # must swallow Ctrl+C, no traceback
    finally:
        srv.ThreadingHTTPServer, srv.webbrowser.open = orig_server, orig_open


# --- v0.4: see it think (live mode) ---

def test_run_forward_captures_per_layer_activations():
    import torch
    from nnviz.live import run_forward
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(1, 4)
    r = run_forward(model, x)

    acts = {a["id"]: a["act"] for a in r["acts"]}
    assert list(acts) == ["in_0", "0", "1", "2"]     # matches graph node ids
    assert [len(v) for v in acts.values()] == [4, 8, 8, 2]
    # captured activations match a manual forward pass (5-decimal rounding)
    h = model[1](model[0](x)).detach().numpy().ravel()
    assert abs(acts["1"][0] - float(h[0])) < 1e-4


def test_run_forward_output_logits_and_probs():
    import torch
    from nnviz.live import run_forward
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    r = run_forward(model, torch.randn(1, 4))
    out = r["output"]
    assert len(out["logits"]) == 3
    assert abs(sum(out["probs"]) - 1.0) < 1e-6
    assert all(p >= 0 for p in out["probs"])
    assert out["pred"] == max(range(3), key=lambda i: out["probs"][i])


def test_conv_activations_reduce_per_channel():
    import torch
    from nnviz.live import run_forward
    model = nn.Sequential(nn.Conv2d(3, 5, 3), nn.ReLU())
    r = run_forward(model, torch.randn(1, 3, 10, 10))
    acts = {a["id"]: len(a["act"]) for a in r["acts"]}
    assert acts == {"in_0": 3, "0": 5, "1": 5}       # one value per channel


def test_huge_activations_capped_on_the_wire():
    import torch
    from nnviz.live import run_forward, MAX_ACT
    model = nn.Sequential(nn.Linear(64, MAX_ACT * 3), nn.Flatten())
    r = run_forward(model, torch.randn(1, 64))
    flat = next(a for a in r["acts"] if a["id"] == "1")
    assert len(flat["act"]) <= MAX_ACT


def test_find_input_shape_probes_until_forward_works():
    import torch
    from nnviz.live import find_input_shape
    mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    assert find_input_shape(mlp) == (1, 4)
    # conv net whose Flatten only fits 32x32 inputs
    cnn = nn.Sequential(
        nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(),
        nn.Linear(4 * 30 * 30, 10))
    assert find_input_shape(cnn) == (1, 3, 32, 32)


def test_image_batch_resizes_and_scales():
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("SKIP image_batch (no pillow)")
        return
    import io
    from PIL import Image
    from nnviz.live import image_batch
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, "PNG")
    x = image_batch((1, 3, 8, 8), buf.getvalue())
    assert tuple(x.shape) == (1, 3, 8, 8)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    assert float(x[0, 0, 0, 0]) > 0.9                # red survives the resize


def test_parse_retains_module_then_run_serves_activations():
    import io
    import json
    import threading
    import socket
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    import torch
    import nnviz.server as srv

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    buf = io.BytesIO()
    torch.save(model, buf)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    real_open = srv.webbrowser.open
    srv.webbrowser.open = lambda *a, **k: None
    httpd = srv.ThreadingHTTPServer(("127.0.0.1", port), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        # drop the file: the module is retained for live runs
        resp = urlopen(Request(f"http://127.0.0.1:{port}/parse", data=buf.getvalue(),
                               headers={"Content-Type": "application/octet-stream"}), timeout=10)
        assert "graph" in json.loads(resp.read())

        body = json.loads(urlopen(f"http://127.0.0.1:{port}/run", data=b"", timeout=10).read())
        assert [a["id"] for a in body["acts"]] == ["in_0", "0", "1"]
        assert abs(sum(body["output"]["probs"]) - 1.0) < 1e-6

        # with no loaded model, /run says so clearly
        srv._STATE["model"] = None
        try:
            urlopen(f"http://127.0.0.1:{port}/run", data=b"", timeout=10)
            raise AssertionError("expected 400")
        except HTTPError as e:
            assert e.code == 400
            assert "nn.Module" in e.read().decode()
    finally:
        srv._STATE["model"] = None
        httpd.shutdown()
        srv.webbrowser.open = real_open


def test_cli_live_loads_model_and_serves():
    import torch
    import nnviz.cli as cli
    import nnviz.server as server

    calls = {}
    orig_serve, orig_state = server.serve, dict(server._STATE)
    server.serve = lambda port=8000: calls.setdefault("port", port)
    torch.save(nn.Sequential(nn.Linear(4, 2)), Path("tmp_live.pth"))
    try:
        cli.main(["live", "tmp_live.pth", "--port", "9123"])
        assert calls.get("port") == 9123
        assert server._STATE["model"] is not None
        assert any(n.op == "Linear" for n in server._STATE["graph"].nodes)
    finally:
        server.serve = orig_serve
        server._STATE.clear()
        server._STATE.update(orig_state)
        Path("tmp_live.pth").unlink()


# --- v0.5: transformers and LLMs ---

def _write_safetensors(path, tensors: dict) -> None:
    """Minimal safetensors writer: u64 header len + JSON header + raw data.
    Accepts f4 (F32), f2 (F16), u2 (BF16 bits), i4 (I32) arrays."""
    import json
    import struct
    import numpy as np
    dt = {"f4": "F32", "f2": "F16", "u2": "BF16", "i4": "I32"}
    header, blob, off = {}, b"", 0
    for name, arr in tensors.items():
        code = arr.dtype.str[1:]          # "<f4" -> "f4"
        n = arr.size * arr.dtype.itemsize
        header[name] = {"dtype": dt[code], "shape": list(arr.shape),
                        "data_offsets": [off, off + n]}
        blob += arr.tobytes()
        off += n
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + blob)


def _tiny_hf_dir() -> Path:
    """A tiny Llama-style HF model: 2 blocks, hidden 8, 2 heads, vocab 16."""
    import json
    import numpy as np
    rng = np.random.default_rng(0).standard_normal
    t = {"model.embed_tokens.weight": rng((16, 8)).astype("f4")}
    for b in range(2):
        p = f"model.layers.{b}"
        t |= {f"{p}.input_layernorm.weight": rng(8).astype("f4"),
              f"{p}.self_attn.q_proj.weight": rng((8, 8)).astype("f4"),
              f"{p}.self_attn.k_proj.weight": rng((8, 8)).astype("f4"),
              f"{p}.self_attn.v_proj.weight": rng((8, 8)).astype("f4"),
              f"{p}.self_attn.o_proj.weight": rng((8, 8)).astype("f4"),
              f"{p}.post_attention_layernorm.weight": rng(8).astype("f4"),
              f"{p}.mlp.gate_proj.weight": rng((16, 8)).astype("f4"),
              f"{p}.mlp.down_proj.weight": rng((8, 16)).astype("f4")}
    t |= {"model.norm.weight": rng(8).astype("f4"),
          "lm_head.weight": rng((16, 8)).astype("f4")}
    d = Path("tmp_hf_tiny")
    d.mkdir(exist_ok=True)
    _write_safetensors(d / "model.safetensors", t)
    (d / "config.json").write_text(json.dumps(
        {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 2,
         "num_attention_heads": 2, "intermediate_size": 16, "vocab_size": 16}))
    return d


def test_detect_hf_directory_and_safetensors():
    from nnviz.detect import detect_format
    d = _tiny_hf_dir()
    try:
        assert detect_format(d) == "hf"
        assert detect_format(d / "model.safetensors") == "hf"
    finally:
        shutil.rmtree(d)


def test_hf_parser_builds_blocks_in_order():
    import shutil
    from nnviz.parsers import parse_hf
    d = _tiny_hf_dir()
    try:
        g = parse_hf(d)
        layers = [n for n in g.nodes if n.kind == "layer"]
        ops = [n.op for n in layers]
        # embedding, then 2 repeated blocks, then final norm + lm head
        assert ops[0] == "Embedding"
        assert ops[-2:] == ["LayerNorm", "Linear"]
        block_ops = [op for op in ops[1:-2]]
        expect_one = ["LayerNorm", "Linear", "Linear", "Linear", "Linear",
                      "LayerNorm", "Linear", "Linear"]
        assert block_ops == expect_one * 2
        # blocks carry their index for the layout
        assert layers[1].attrs["block"] == 0
        assert layers[len(layers) // 2].attrs["block"] == 1
        assert g.meta["format"] == "huggingface"
        assert g.meta["params"] == 16 * 8 + 2 * (8 + 4 * 64 + 8 + 16 * 8 + 8 * 16) + 8 + 16 * 8
        q = next(n for n in layers if n.id.endswith("q_proj") and n.attrs["block"] == 0)
        assert q.attrs["in_features"] == 8 and q.attrs["out_features"] == 8
    finally:
        shutil.rmtree(d)


def test_hf_parser_attention_heads_and_attends():
    import shutil
    from nnviz.parsers import parse_hf
    d = _tiny_hf_dir()
    try:
        g = parse_hf(d)
        heads = [n for n in g.nodes if n.kind == "head"]
        assert len(heads) == 4                     # 2 heads x 2 blocks
        assert all(h.attrs["dim"] == 4 for h in heads)   # hidden 8 / 2 heads
        attends = [e for e in g.edges if e.kind == "attends"]
        assert len(attends) == 4                   # one per head, into o_proj
        # each head receives q, k and v
        for h in heads:
            ins = [e.src for e in g.edges if e.dst == h.id]
            assert len(ins) == 3
    finally:
        shutil.rmtree(d)


def test_hf_parser_weight_stats_and_cap():
    import shutil
    import nnviz.parsers.hf as hf
    d = _tiny_hf_dir()
    try:
        g = hf.parse_hf(d)
        q = next(n for n in g.nodes if n.id.endswith("q_proj") and n.attrs["block"] == 0)
        assert q.attrs["weights"]["count"] == 64
        # tensors above the stats cap keep their shape but skip the histogram
        real_cap = hf.STAT_CAP
        hf.STAT_CAP = 100                          # bytes: everything is too big now
        try:
            g2 = hf.parse_hf(d)
            q2 = next(n for n in g2.nodes if n.id.endswith("q_proj") and n.attrs["block"] == 0)
            assert "weights" not in q2.attrs
            assert q2.attrs["out_features"] == 8   # shape info survives
        finally:
            hf.STAT_CAP = real_cap
    finally:
        shutil.rmtree(d)


def test_hf_layout_blocks_in_sequence():
    import shutil
    from nnviz.layout import layout
    from nnviz.parsers import parse_hf
    d = _tiny_hf_dir()
    try:
        g = parse_hf(d)
        layout(g)
        layers = [n for n in g.nodes if n.kind == "layer"]
        xs = [n.pos[0] for n in layers]
        assert all(b > a for a, b in zip(xs, xs[1:]))       # strict sequence
        # the gap between blocks is wider than inside a block
        q0 = next(n for n in layers if n.id.endswith("q_proj") and n.attrs["block"] == 0)
        norm0 = next(n for n in layers if n.id.endswith("input_layernorm") and n.attrs["block"] == 0)
        o0 = next(n for n in layers if n.id.endswith("o_proj") and n.attrs["block"] == 0)
        q1 = next(n for n in layers if n.id.endswith("q_proj") and n.attrs["block"] == 1)
        within = o0.pos[0] - norm0.pos[0] if o0.pos[0] > norm0.pos[0] else q0.pos[0] - norm0.pos[0]
        between = q1.pos[0] - o0.pos[0]
        assert between > within
        # heads sit below their parent layer, spread apart
        heads = [n for n in g.nodes if n.kind == "head" and n.attrs.get("block") == 0]
        assert all(h.pos[1] < 0 for h in heads)
        assert heads[0].pos[2] != heads[1].pos[2]
    finally:
        shutil.rmtree(d)


def test_classify_detects_transformer_from_heads():
    from nnviz.detect import classify
    from nnviz.ir import Graph, Node
    g = Graph()
    g.nodes = [Node("0", "layer", "LayerNorm"), Node("1", "layer", "Linear"),
               Node("1.h0", "head", "Head")]
    assert classify(g) == "Transformer"


def test_cli_load_hf_dir_writes_html():
    import nnviz.cli as cli
    d = _tiny_hf_dir()
    try:
        cli.main(["load", str(d), "-o", "tmp_hf_out.html"])
        html = Path("tmp_hf_out.html").read_text(encoding="utf-8")
        assert "__MODEL_DATA__" not in html
        assert "self_attn" in html                      # graph embedded
    finally:
        shutil.rmtree(d)
        Path("tmp_hf_out.html").unlink(missing_ok=True)


def test_hf_parser_reads_packed_int4_and_text_config():
    """INT4 checkpoints: weights come as weight_packed (+scale/shape),
    the head count hides in config.text_config, vision tensors wait for v0.6."""
    import json
    import numpy as np
    from nnviz.parsers import parse_hf
    d = Path("tmp_hf_packed")
    d.mkdir(exist_ok=True)
    i32 = lambda a: np.asarray(a, dtype="<i4")
    u2 = lambda a: np.asarray(a, dtype="<u2")          # BF16 bits
    t = {"model.language_model.embed_tokens.weight": u2(np.zeros((16, 8), np.uint32) >> 16),
         "model.language_model.layers.0.input_layernorm.weight": u2(np.full(8, 0x3F80)),
         "model.language_model.layers.0.self_attn.q_proj.weight_packed": i32(np.zeros((8, 1))),
         "model.language_model.layers.0.self_attn.q_proj.weight_scale": u2(np.zeros((8, 1))),
         "model.language_model.layers.0.self_attn.o_proj.weight_packed": i32(np.zeros((8, 1))),
         "model.visual.blocks.0.attn.qkv.weight_packed": i32(np.zeros((8, 1))),
         "lm_head.weight": u2(np.zeros((16, 8), np.uint32) >> 16)}
    _write_safetensors(d / "model.safetensors", t)
    (d / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5", "text_config": {"num_attention_heads": 2}}))
    try:
        g = parse_hf(d)
        ids = [n.id for n in g.nodes]
        # packed weight becomes the node; scale/shape and visual stay out
        assert "model.language_model.layers.0.self_attn.q_proj" in ids
        assert "model.visual.blocks.0.attn.qkv" not in ids
        assert not any("weight_scale" in i for i in ids)
        q = next(n for n in g.nodes if n.id.endswith("q_proj"))
        assert q.attrs["in_features"] == 8 and q.attrs["out_features"] == 8   # 1 x 8-per-int32
        assert "weights" not in q.attrs            # packed ints: no honest histogram
        # heads from text_config
        assert len([n for n in g.nodes if n.kind == "head"]) == 2
        assert g.meta["params"] == 16 * 8 + 8 + 8 * 8 + 8 * 8 + 16 * 8    # packed counts x8
    finally:
        shutil.rmtree(d)


def test_hf_parser_bfloat16_stats():
    import numpy as np
    import nnviz.parsers.hf as hf
    from nnviz.parsers import parse_hf
    d = Path("tmp_hf_bf16")
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(1).standard_normal
    w = rng((4, 4)).astype("f4")
    bits = (w.view("<u4") >> 16).astype("<u2")     # f32 -> bf16 bits
    t = {"model.layers.0.mlp.gate_proj.weight": bits}
    _write_safetensors(d / "model.safetensors", t)
    try:
        g = parse_hf(d)
        gate = next(n for n in g.nodes if n.op == "Linear")
        ws = gate.attrs["weights"]
        assert ws["count"] == 16
        assert abs(ws["mean"] - float(w.mean())) < 0.02    # bf16 keeps ~3 digits
    finally:
        shutil.rmtree(d)


def test_server_load_local_folder_path():
    """/load parses a model from a local path: no upload, no size limit,
    config.json included, so head nodes appear."""
    import json
    import threading
    import socket
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    import nnviz.server as srv

    d = _tiny_hf_dir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    real_open = srv.webbrowser.open
    srv.webbrowser.open = lambda *a, **k: None
    httpd = srv.ThreadingHTTPServer(("127.0.0.1", port), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        body = json.dumps({"path": str(d)}).encode()
        resp = urlopen(Request(f"http://127.0.0.1:{port}/load", data=body,
                               headers={"Content-Type": "application/json"}), timeout=10)
        g = json.loads(resp.read())["graph"]
        assert any(n["kind"] == "head" for n in g["nodes"])       # config.json was read
        assert any(n["op"] == "Embedding" for n in g["nodes"])

        bad = json.dumps({"path": "Z:/does/not/exist"}).encode()
        try:
            urlopen(Request(f"http://127.0.0.1:{port}/load", data=bad,
                            headers={"Content-Type": "application/json"}), timeout=10)
            raise AssertionError("expected 400")
        except HTTPError as e:
            assert e.code == 400
            assert "error" in e.read().decode()
    finally:
        srv._STATE["model"] = None
        srv._STATE["graph"] = None
        httpd.shutdown()
        shutil.rmtree(d)


def test_server_parses_safetensors_upload():
    import json
    import threading
    import socket
    import shutil
    from urllib.request import urlopen, Request
    import numpy as np
    import nnviz.server as srv

    d = _tiny_hf_dir()
    blob = (d / "model.safetensors").read_bytes()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    real_open = srv.webbrowser.open
    srv.webbrowser.open = lambda *a, **k: None
    httpd = srv.ThreadingHTTPServer(("127.0.0.1", port), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        resp = urlopen(Request(f"http://127.0.0.1:{port}/parse", data=blob,
                               headers={"Content-Type": "application/octet-stream"}), timeout=10)
        g = json.loads(resp.read())["graph"]
        assert any(n["op"] == "Embedding" for n in g["nodes"])
        # no config.json in a bare upload: the head count is unknown, so
        # no head nodes. Use /load with a folder path for the full graph.
        assert not any(n["kind"] == "head" for n in g["nodes"])
        assert any("q_proj" in n["id"] for n in g["nodes"])
    finally:
        srv._STATE["model"] = None
        srv._STATE["graph"] = None
        httpd.shutdown()
        shutil.rmtree(d)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
