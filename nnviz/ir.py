"""Unified intermediate representation: typed nodes + edges, JSON-serializable."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    id: str
    kind: str                 # "layer" for now; "attention_head" etc. later
    op: str                   # Linear, Conv2d, ReLU, Input, ...
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "op": self.op,
                "pos": self.pos, "attrs": self.attrs}


@dataclass
class Edge:
    src: str
    dst: str
    kind: str = "connects"

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "kind": self.kind}


@dataclass
class Graph:
    meta: dict[str, Any] = field(default_factory=dict)
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"meta": self.meta,
                "nodes": [n.to_dict() for n in self.nodes],
                "edges": [e.to_dict() for e in self.edges]}
