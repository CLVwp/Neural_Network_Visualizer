"""3D layout: layers as columns along X, neuron counts for the viewer."""


def layout(g) -> None:
    layers = [n for n in g.nodes if n.kind == "layer"]
    spacing = 14.0  # room for neuron grids between columns
    for i, n in enumerate(layers):
        n.pos = [i * spacing, 0.0, 0.0]
