"""ONNX parser: graph proto -> IR.

ONNX graphs are explicit DAGs, so edges follow tensor provenance. Branched
nets come out correct here (unlike the definition-order torch parser).
"""
import onnx
from onnx import numpy_helper

from ..ir import Edge, Graph, Node
from ..stats import apply_strengths, weight_stats

_DENSE = ("Gemm", "MatMul")


def _weights(graph) -> dict[str, object]:
    return {t.name: numpy_helper.to_array(t) for t in graph.initializer}


def parse_onnx(model) -> Graph:
    if isinstance(model, (str, bytes)):
        model = onnx.load(model)
    graph = model.graph
    prod = {out: n for n in graph.node for out in n.output}
    weights = _weights(graph)
    graph_inputs = [i.name for i in graph.input if i.name not in weights]

    g = Graph(meta={"format": "onnx", "params": sum(w.size for w in weights.values())})
    ids: dict[str, str] = {}          # node output -> IR node id
    size_of: dict[str, int] = {}      # tensor -> running unit count

    # input entity: where data enters
    in_name = graph_inputs[0] if graph_inputs else ""
    dims = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim] if in_name else []
    in_size = dims[1] if len(dims) >= 4 else (dims[-1] if dims else 1)
    g.nodes.append(Node("in_0", "layer", "Input", attrs={"size": in_size}))
    size_of[in_name] = in_size

    for i, n in enumerate(graph.node):
        nid = f"{n.op_type}_{i}"
        a: dict = {}
        w = next((weights[name] for name in n.input if name in weights), None)
        if w is not None:
            a["weights"] = weight_stats(w)
        if n.op_type == "Conv" and w is not None:
            a |= {"out_channels": int(w.shape[0]), "in_channels": int(w.shape[1]),
                  "kernel": list(w.shape[2:]), "size": int(w.shape[0])}
        elif n.op_type in _DENSE and w is not None and w.ndim == 2:
            # ponytail: assumes [out, in] (Gemm convention); transposed MatMul
            # weights give swapped counts until shape inference lands
            out_f, in_f = (int(w.shape[0]), int(w.shape[1])) if n.op_type == "Gemm" else (int(w.shape[1]), int(w.shape[0]))
            a |= {"in_features": in_f, "out_features": out_f, "size": out_f}
        else:
            a["size"] = next((size_of[t] for t in n.input if t in size_of), 0)
        if n.output:
            size_of[n.output[0]] = a["size"]
        g.nodes.append(Node(nid, "layer", n.op_type, attrs=a))
        ids[n.output[0]] = nid

        for t in n.input:
            if t in prod:                       # produced by a previous node
                g.edges.append(Edge(ids[prod[t].output[0]], nid))
            elif t in graph_inputs:             # fed by the graph input
                g.edges.append(Edge("in_0", nid))
    apply_strengths(g)
    return g
