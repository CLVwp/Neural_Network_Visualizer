# Neural Network Visualizer

Explore any neural network as an interactive 3D graph in your browser.
Import a model file in one click. The tool detects the format and the model type.
It builds the graph. You rotate, zoom, and inspect every neuron and connection.

Supports CNNs, MLPs, Transformers, LLMs, and VLMs.

**Status: design phase.** The roadmap below shows the full plan.

## Why

Model architecture lives in code and tensors. Standard tools show it as flat text
or 2D diagrams. Structure in 3D is easier to read. Big models become navigable
spaces instead of walls of shapes. You can literally see a network think.

## How it works

```
model file (.pth / .onnx / .safetensors / .keras)
        │
        ▼
┌─────────────────────┐
│  Format detector    │  magic bytes + extension → parser
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Parser             │  framework-specific → unified graph
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  IR (JSON graph)    │  typed entities + typed relations
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  3D viewer (WebGL)  │  Three.js, runs in the browser
└─────────────────────┘
```

The IR is the core idea. Every framework maps to one graph schema.
The viewer only knows the IR. A new format needs one new parser, nothing else.

## Entities and relations

The IR models two kinds of objects.

Entities:

| Entity         | Meaning                                        |
| -------------- | ---------------------------------------------- |
| `Layer`        | One computation layer (conv, dense, attention) |
| `Neuron`       | One unit inside a layer                        |
| `Tensor`       | A data tensor that flows between layers        |
| `AttentionHead`| One attention head inside a transformer block  |
| `Branch`       | A model branch (vision tower, language tower)  |

Relations:

| Relation    | Meaning                                  |
| ----------- | ---------------------------------------- |
| `connects`  | A layer feeds another layer              |
| `attends`   | An attention head attends to positions   |
| `contains`  | A layer contains neurons or heads        |
| `flows`     | A tensor flows along a connection        |

Each entity carries metadata: shape, parameter count, activation stats.

## Visualization modes

**Static mode.** Load a file. See the architecture. Edge thickness shows weight
magnitude. Node color shows layer type. No GPU and no inference required.

**Live mode.** Connect the viewer to a running model. Send an input. Watch
activations propagate through the graph in real time. This is the "see it think"
mode.

## Supported formats

| Format                  | Files                              | Status  |
| ----------------------- | ---------------------------------- | ------- |
| PyTorch                 | `.pt`, `.pth`, `state_dict`        | v0.1    |
| ONNX                    | `.onnx`                            | v0.2    |
| HuggingFace             | `safetensors`, `config.json`       | v0.5    |
| TensorFlow / Keras      | `.keras`, SavedModel               | v0.6    |

Format detection reads the file signature. You never pick a parser by hand.

## Roadmap

### v0.1 — First light

- [ ] Python package skeleton (`nnviz`)
- [ ] PyTorch parser: MLP and CNN (`nn.Sequential`, `state_dict`)
- [ ] IR schema: entities, relations, metadata (JSON)
- [ ] Layout algorithm: layers as columns, neurons as points
- [ ] 3D viewer: static graph, orbit controls, click to inspect a node
- [ ] CLI: `nnviz load model.pth` → `model.json` → open in browser

### v0.2 — Import anything

- [ ] Format auto-detection by file signature
- [ ] ONNX parser (graph proto → IR)
- [ ] Drag and drop a file onto the web page
- [ ] Model type classifier: MLP / CNN / Transformer from layer statistics
- [ ] Error reports for unsupported or corrupted files

### v0.3 — Make weights visible

- [ ] Edge thickness and color mapped to weight magnitude
- [ ] Weight distributions on node click (histogram panel)
- [ ] Layout per layer type: CNN feature maps as 3D grids, dense layers as walls
- [ ] Layer collapse and expand (hide neurons, keep the layer node)
- [ ] Color themes and legend

### v0.4 — See it think (live mode)

- [ ] Local server (FastAPI) that loads a model
- [ ] WebSocket stream of activations during a forward pass
- [ ] Animated activation wave through the graph
- [ ] Input picker: image for a CNN, text tokens for a transformer
- [ ] Output panel: logits, class probabilities

### v0.5 — Transformers and LLMs

- [ ] HuggingFace `safetensors` + `config.json` parser
- [ ] Transformer block layout: repeated blocks in a spatial sequence
- [ ] Attention heads as subgraphs with `attends` relations
- [ ] Token flow view: one token path highlighted through all blocks
- [ ] Handle 1B+ parameter models: layer-of-detail (show layers first, neurons on demand)

### v0.6 — VLMs and multi-modal models

- [ ] Branch layout: vision tower and language tower as separate clusters
- [ ] Cross-attention links between modalities
- [ ] Keras / TensorFlow parser
- [ ] Modality filter: show vision only, language only, or both

### v0.7 — Training dynamics

- [ ] Gradient visualization during training
- [ ] Loss curve panel next to the graph
- [ ] Time travel: step through training checkpoints
- [ ] Weight change diff between two checkpoints (what moved)

### v0.8 — Scale

- [ ] Instanced rendering for millions of edges
- [ ] Level of detail: neurons fade out before layers do
- [ ] Web-worker parsing for big files
- [ ] Performance target: 60 fps on a 100M parameter model on a laptop GPU

### v1.0 — Ship it

- [ ] Hosted demo with example models (one click, no install)
- [ ] Documentation site
- [ ] Plugin guide: write a parser for your own format
- [ ] Stable IR schema v1

### Later — ideas parking lot

- VR mode (WebXR)
- Side-by-side comparison of two models
- Video export of an activation wave
- Community layout gallery

## Project layout (planned)

```
nnviz/
  detect.py      # format detection
  parsers/       # torch.py, onnx.py, hf.py, keras.py
  ir.py          # IR schema + validation
  layout.py      # 3D layout algorithms
  server.py      # live mode (FastAPI)
web/
  viewer/        # Three.js viewer
```

## Getting started

Requires Python 3.10 or later.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
nnviz model.pth             # writes model.html. Open it in a browser.
```

`nnviz` accepts a pickled `nn.Module` or a `state_dict`. The output is one
self-contained HTML file. No server and no install needed to view it.

## License

MIT.
