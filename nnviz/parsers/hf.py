"""HuggingFace parser: config.json + safetensors -> IR graph.

Reads the safetensors header (u64 length + JSON) for names and shapes.
Never loads the whole checkpoint: 1B+ models stay cheap to open.
Tensor data is read per tensor, only under STAT_CAP, for weight stats.
"""
import json
import re
import struct
from pathlib import Path

import numpy as np

from ..ir import Edge, Graph, Node
from ..stats import weight_stats

STAT_CAP = 128 * 1024 * 1024   # ponytail: histograms above 128 MiB are skipped; LOD in v0.8
HEAD_CAP = 64                  # ponytail: head nodes capped; bigger head counts stay attrs
PACK = 8                       # ponytail: int4 packing assumes 8 values per int32

_BLOCK_RE = re.compile(r"(?:^|\.)(?:h|layers?|blocks?)\.(\d+)\.")

# one name pattern list per role, checked in this order
_ROLES = [
    ("attn_o",  ("o_proj", "out_proj", "wo", "attn.c_proj", "attention.c_proj")),
    ("attn_q",  ("q_proj", "query", "wq", "qkv", "c_attn", "in_proj_qkv")),
    ("attn_k",  ("k_proj", "key", "wk")),
    ("attn_v",  ("v_proj", "value", "wv")),
    ("mlp_in",  ("gate_proj", "up_proj", "c_fc", "w1", "w3", "h_to_4h", "fc1", "wi")),
    ("mlp_out", ("down_proj", "c_proj", "w2", "4h_to_h", "fc2")),
    ("lm_head", ("lm_head", "output.")),
    ("embed",   ("embed", "wte", "tok_embeddings", "emb")),
]

_DTYPES = {"F32": "<f4", "F16": "<f2", "BF16": "<u2", "I64": "<i8", "I32": "<i4",
           "I16": "<i2", "I8": "|i1", "U8": "|u1", "F64": "<f8", "BOOL": "|b1"}
_FLOATS = {"F32", "F16", "BF16", "F64"}


def _read_header(source):
    """safetensors bytes or path -> (header dict, data start offset)."""
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        return header, 8 + hlen
    (hlen,) = struct.unpack_from("<Q", source)
    header = json.loads(source[8:8 + hlen])
    return header, 8 + hlen


def _tensor(header, source, data_start, name):
    """One tensor as a numpy view (mmap for files, view for bytes)."""
    m = header[name]
    off = data_start + m["data_offsets"][0]
    shape = tuple(m["shape"])
    dt = _DTYPES.get(m.get("dtype"), "<f4")
    if isinstance(source, (str, Path)):
        a = np.memmap(source, dtype=dt, mode="r", offset=off, shape=shape)
    else:
        n = int(np.prod(shape)) if shape else 1
        a = np.frombuffer(source, dtype=dt, count=n, offset=off).reshape(shape)
    if m.get("dtype") == "BF16":               # bf16 bits -> f32 values
        return (a.astype("<u4") << 16).view("<f4")
    return a


def _role(name: str, in_block: bool):
    """Tensor name -> (role, slot). None means: no node for this tensor."""
    low = name.lower()
    if "norm" in low or ".ln" in low or low.startswith("ln"):
        if in_block:
            # norms inside attention (q_norm, k_norm, linear_attn.norm) are
            # internal details: skip them, the projection chain stays clean
            return None if "attn" in low else ("norm", 0)
        return ("final_norm", 9)
    for role, keys in _ROLES:
        if any(k in low for k in keys):
            if role == "embed":
                return ("embed", -1) if not in_block else None
            return (role, {"attn_q": 1, "attn_k": 2, "attn_v": 3,
                           "attn_o": 5, "mlp_in": 7, "mlp_out": 8,
                           "lm_head": 10}[role])
    return None


def _config_heads(config) -> int:
    # multimodal configs nest the text stack under text_config
    for cfg in (config, (config or {}).get("text_config")):
        for k in ("num_attention_heads", "n_head", "num_heads", "n_heads"):
            if cfg and cfg.get(k):
                return int(cfg[k])
    return 0


def parse_hf(source, config: dict | None = None) -> Graph:
    """safetensors path/bytes/model dir (+ optional config.json) -> IR graph."""
    if isinstance(source, (str, Path)):
        base = Path(source)
        if base.is_dir():                      # HF folder: pick the weights inside
            source = next(base.glob("*.safetensors"), None)
            if source is None:
                raise ValueError(f"no .safetensors file in {base}")
        else:
            base = base.parent
        if config is None and (base / "config.json").exists():
            config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    header, data_start = _read_header(source)
    # a weight is ".weight" or an int4 ".weight_packed"; scales/shapes/bias
    # are auxiliary. Vision towers wait for v0.6 (branch layout).
    weights = {}
    for n, m in header.items():
        if ".visual." in n or not m.get("shape"):
            continue
        for suffix in (".weight", ".weight_packed"):
            if n.endswith(suffix):
                weights[n[:-len(suffix)]] = (m, suffix == ".weight_packed")

    heads_n = _config_heads(config)

    def nparams(m, packed):
        return int(np.prod(m["shape"])) * (PACK if packed else 1)

    g = Graph(meta={"format": "huggingface",
                    "params": sum(nparams(m, p) for m, p in weights.values()),
                    "num_heads": heads_n})

    entries = []                       # (sort key, id, role, block, shape, packed)
    for name, (m, packed) in weights.items():
        hit = _BLOCK_RE.search(name)
        block = int(hit.group(1)) if hit else None
        role = _role(name, in_block=block is not None)
        if role is None:
            continue
        r, slot = role
        # embed sorts before all blocks; final norm and lm head after
        bkey = block if block is not None else ((1 << 30) if slot >= 0 else -1)
        entries.append(((bkey, slot, name), name, r, block, tuple(m["shape"]), packed))
    # the first norm of a block opens it; later norms sit between attn and MLP
    norm_names: dict = {}
    for k, name, r, block, shape, packed in entries:
        if r == "norm" and block is not None:
            norm_names.setdefault(block, []).append(name)
    entries = [((k[0], 6, name), name, r, block, shape, packed)
               if r == "norm" and block is not None
               and norm_names[block].index(name) > 0 else (k, name, r, block, shape, packed)
               for k, name, r, block, shape, packed in entries]
    entries.sort(key=lambda e: e[0])
    shapes = {name: shape for _, name, _, _, shape, _ in entries}
    hidden = next((s[1] for s in shapes.values() if len(s) == 2), 0)
    g.meta["hidden_size"] = hidden

    by_block: dict = {}
    for _, name, r, block, shape, packed in entries:
        a = {"shape": list(shape), "block": block}
        if packed:                     # packed shape is [out, in/8]: show true sizes
            shape = (shape[0], shape[1] * PACK) if len(shape) == 2 else shape
            a["quant"] = "int4-packed"
        if r == "embed":
            op, a["vocab"], a["size"] = "Embedding", shape[0], shape[1]
        elif r == "final_norm" or r == "norm":
            op, a["size"] = "LayerNorm", shape[0]
        elif len(shape) == 2:
            op = "Linear"
            a |= {"out_features": shape[0], "in_features": shape[1], "size": shape[0]}
        else:
            op, a["size"] = "Linear", shape[0]
        node = Node(name, "layer", op, attrs=a)
        if len(shape) == 2 and not packed:
            m, _ = weights[name]
            if m.get("dtype") in _FLOATS:      # packed ints have no honest histogram
                t = _tensor(header, source, data_start, name + ".weight")
                if t.nbytes <= STAT_CAP:
                    a["weights"] = weight_stats(t)
                if isinstance(t, np.memmap):
                    t._mmap.close()
                del t
        g.nodes.append(node)
        by_block.setdefault(block, []).append((r, node))

    # wiring: blocks chain into each other; inside a block q/k/v feed the
    # heads, the heads feed o_proj (kind "attends"), then the MLP tail.
    blocks = sorted(b for b in by_block if b is not None)
    prev_exit = next((n.id for r, n in by_block.get(None, []) if r == "embed"), None)
    for b in blocks:
        items = by_block[b]
        pick = lambda r: next((n for rr, n in items if rr == r), None)
        first = pick("norm") or pick("attn_q") or items[0][1]
        norms = [n for r, n in items if r == "norm"]
        qs = [n for r, n in items if r in ("attn_q", "attn_k", "attn_v")]
        o = pick("attn_o")
        if prev_exit:
            g.edges.append(Edge(prev_exit, first.id))
        for q in qs:
            if q is not first:
                g.edges.append(Edge(first.id, q.id))
        if heads_n and 1 < heads_n <= HEAD_CAP and o is not None:
            for i in range(heads_n):
                h = Node(f"{o.id}.h{i}", "head", "Head",
                         attrs={"head": i, "heads": heads_n,
                                "dim": hidden // heads_n, "block": b})
                g.nodes.append(h)
                for q in qs:
                    g.edges.append(Edge(q.id, h.id))
                g.edges.append(Edge(h.id, o.id, "attends"))
        elif o:
            for q in qs:                   # no head count known: wire straight through
                g.edges.append(Edge(q.id, o.id))
        elif qs and len(qs) > 1:           # attention with no output proj seen
            for q in qs[1:]:
                g.edges.append(Edge(qs[0].id, q.id))
        attn_exit = o or (qs[-1] if qs else first)
        tail = [pick("mlp_in"), pick("mlp_out")]
        chain = ([norms[1]] if len(norms) > 1 else []) + [t for t in tail if t]
        for a2, b2 in zip([attn_exit] + chain, chain):
            g.edges.append(Edge(a2.id, b2.id))
        prev_exit = chain[-1].id if chain else attn_exit.id
    for r, n in by_block.get(None, []):
        if r == "final_norm" and prev_exit:
            g.edges.append(Edge(prev_exit, n.id))
            prev_exit = n.id
        elif r == "lm_head" and prev_exit:
            g.edges.append(Edge(prev_exit, n.id))
            prev_exit = n.id
    if not g.edges:                        # unknown naming: fall back to a chain
        ids = [n.id for n in g.nodes if n.kind == "layer"]
        g.edges = [Edge(s, d) for s, d in zip(ids, ids[1:])]
    return g
