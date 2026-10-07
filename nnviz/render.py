"""Inject a graph into the viewer template: one self-contained HTML file."""
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("viewer.html")


def render_html(g) -> str:
    payload = json.dumps(g.to_dict()).replace("</", "<\\/")  # keep </script> out of the payload
    return TEMPLATE.read_text(encoding="utf-8").replace("__MODEL_DATA__", payload)
