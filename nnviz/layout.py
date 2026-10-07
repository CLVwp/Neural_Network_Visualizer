"""3D layout: layers as columns along X, blocks as a spatial sequence."""


def layout(g, spacing: float = 10.0, block_gap: float = 8.0) -> None:
    """Blocks repeat along X with a wider gap between them. Attention heads
    hang below their parent layer, spread along Z."""
    pos_of = {}
    x = 0.0
    prev_block = None
    for n in g.nodes:
        if n.kind not in ("layer", "head"):
            continue
        if n.kind == "head":
            parent = n.id.rsplit(".h", 1)[0]
            px = pos_of.get(parent, [0.0, 0.0, 0.0])[0]
            i, total = n.attrs.get("head", 0), max(1, n.attrs.get("heads", 1))
            n.pos = [px, -7.0, (i - (total - 1) / 2) * 2.2]
            continue
        b = n.attrs.get("block")
        if pos_of:
            x += spacing + (block_gap if b != prev_block else 0.0)
        n.pos = [x, 0.0, 0.0]
        pos_of[n.id] = n.pos
        prev_block = b
