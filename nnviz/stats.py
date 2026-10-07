"""Weight statistics: numbers on nodes, strength on edges."""
import numpy as np

BINS = 40


def weight_stats(w) -> dict:
    """Histogram + summary of one weight tensor. Accepts torch or numpy."""
    x = w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
    n_out = x.shape[0]                    # output neurons (conv: filters)
    x = x.astype(np.float64).ravel()
    hist, edges = np.histogram(x, bins=BINS)
    out = {"count": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
           "min": float(x.min()), "max": float(x.max()),
           "mean_abs": float(np.abs(x).mean()),
           "hist": [int(v) for v in hist],
           "hist_lo": float(edges[0]), "hist_hi": float(edges[-1])}
    # per-neuron incoming strength: mean |w| of each output unit's row
    if n_out <= 8192:   # ponytail: cap neuron detail; LOD arrives in v0.8
        rows = np.abs(x).reshape(n_out, -1).mean(axis=1)
        out["row_strengths"] = [round(float(v), 5) for v in rows]
    return out


def apply_strengths(g) -> None:
    """Edge strength = incoming weights of dst, else outgoing weights of src."""
    by_id = {n.id: n for n in g.nodes}
    for e in g.edges:
        s = (by_id[e.dst].attrs.get("weights") or by_id[e.src].attrs.get("weights"))
        if s:
            e.attrs["strength"] = s["mean_abs"]
