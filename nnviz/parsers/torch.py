"""PyTorch parser: nn.Module -> IR graph.

# ponytail: layers are chained in definition order (named_modules). This is
# execution order for Sequential/MLP/simple CNNs but wrong for branched nets
# (ResNet skip connections). Hook-based tracing arrives with live mode (v0.4).
"""
import torch.nn as nn

from ..ir import Edge, Graph, Node


def _attrs(mod: nn.Module) -> dict:
    a: dict = {}
    if isinstance(mod, nn.Linear):
        a |= {"in_features": mod.in_features, "out_features": mod.out_features}
    elif isinstance(mod, nn.Conv2d):
        a |= {"in_channels": mod.in_channels, "out_channels": mod.out_channels,
              "kernel": list(mod.kernel_size)}
    return a


def parse_module(model: nn.Module) -> Graph:
    # leaf computation modules only (skip containers and the root)
    leaves = [(name, m) for name, m in model.named_modules()
              if name and len(list(m.children())) == 0]

    g = Graph(meta={"format": "pytorch"})
    ids: list[str] = []
    running_size = 0
    for i, (name, m) in enumerate(leaves):
        a = _attrs(m)
        op = type(m).__name__
        size = a.get("out_features") or a.get("out_channels") or running_size
        if i == 0:
            # input layer: where data enters, sized by the first layer's input
            running_size = a.get("in_features") or a.get("in_channels") or 1
            g.nodes.append(Node(f"in_0", "layer", "Input", attrs={"size": running_size}))
            ids.append("in_0")
        g.nodes.append(Node(name or f"layer_{i}", "layer", op, attrs=a | {"size": size}))
        ids.append(name or f"layer_{i}")
        running_size = size
    g.edges = [Edge(s, d) for s, d in zip(ids, ids[1:])]
    g.meta["params"] = sum(p.numel() for p in model.parameters())
    return g


def parse_state_dict(sd: dict) -> Graph:
    """state_dict -> IR. Parameter shapes reveal layer sizes; activations
    (ReLU...) have no parameters, so they do not appear here."""
    # group params by module prefix, first-appearance order
    mods: dict[str, dict] = {}
    for key, t in sd.items():
        prefix, _, leaf = key.rpartition(".")
        if leaf == "weight":
            mods.setdefault(prefix or key, t)

    g = Graph(meta={"format": "pytorch-state_dict"})
    ids: list[str] = []
    for i, (name, w) in enumerate(mods.items()):
        if w.dim() == 2:                      # Linear weight: [out, in]
            op, a = "Linear", {"in_features": w.shape[1], "out_features": w.shape[0], "size": w.shape[0]}
        elif w.dim() == 4:                    # Conv2d weight: [out, in, kh, kw]
            op, a = "Conv2d", {"in_channels": w.shape[1], "out_channels": w.shape[0],
                               "kernel": list(w.shape[2:]), "size": w.shape[0]}
        else:
            continue
        if i == 0:
            ins = a["in_features"] if op == "Linear" else a["in_channels"]
            g.nodes.append(Node("in_0", "layer", "Input", attrs={"size": ins}))
            ids.append("in_0")
        g.nodes.append(Node(name, "layer", op, attrs=a))
        ids.append(name)
    g.edges = [Edge(s, d) for s, d in zip(ids, ids[1:])]
    return g
