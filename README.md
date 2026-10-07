# Neural Network Visualizer

Explore any neural network as an interactive 3D graph in your browser.
Import a model file in one click. The tool detects the format and the model type.
It builds the graph. You rotate, zoom, and inspect every neuron and connection.

Supports CNNs, MLPs, Transformers, LLMs, and VLMs.

**Status: working v0.5.** PyTorch, ONNX, and HuggingFace work today. Weights
and activations are visible. Transformers show their blocks and attention
heads. The roadmap shows the full plan.

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
| PyTorch                 | `.pt`, `.pth`, `state_dict`        | **done** |
| ONNX                    | `.onnx`                            | **done** |
| HuggingFace             | `safetensors`, `config.json`       | **done** |
| TensorFlow / Keras      | `.keras`, SavedModel               | v0.6    |

Format detection reads the file signature. You never pick a parser by hand.

## Roadmap

### v0.1 — First light ✅

- [x] Python package skeleton (`nnviz`)
- [x] PyTorch parser: MLP and CNN (`nn.Sequential`, `state_dict`)
- [x] IR schema: entities, relations, metadata (JSON)
- [x] Layout algorithm: layers as columns, neurons as points
- [x] 3D viewer: static graph, orbit controls, click to inspect a node
- [x] CLI: `nnviz model.pth` → self-contained `model.html`

### v0.2 — Import anything ✅

- [x] Format auto-detection by file signature
- [x] ONNX parser (graph proto → IR)
- [x] Drag and drop a file onto the web page (`nnviz serve`)
- [x] Model type classifier: MLP / CNN / Transformer from layer statistics
- [x] Error reports for unsupported or corrupted files

### v0.3 — Make weights visible ✅

- [x] Edge thickness and color mapped to weight magnitude
- [x] Weight distributions on node click (histogram panel)
- [x] Per-neuron weight strength: neuron glow, click for `|w|` and rank
- [x] Truthful wiring: 1:1 beams for pass-through ops, bipartite fans for weighted layers
- [x] Layout per layer type: CNN feature maps as 3D grids, dense layers as walls
- [x] Layer collapse and expand (hide neurons, keep the layer node)
- [x] Selection chain (Shift+click): highlight the neurons and connections you pick
- [x] Color themes and legend

### v0.4 — See it think (live mode) ✅

- [x] Local server that loads a model (stdlib HTTP, no new dependencies)
- [x] Activations of every layer, captured during one forward pass
- [x] Animated activation wave through the graph
- [x] Input picker: random noise or an image (CNN)
- [x] Output panel: logits, class probabilities

### v0.5 — Transformers and LLMs ✅

- [x] HuggingFace `safetensors` + `config.json` parser
- [x] Transformer block layout: repeated blocks in a spatial sequence
- [x] Attention heads as subgraphs with `attends` relations
- [x] Path view: click a neuron, press T. One unit traced through all blocks
- [x] Handle 1B+ parameter models: layer-of-detail. Big models open with
      layers collapsed. Double-click a layer to show its neurons

The parser reads the safetensors header only. It never loads the whole
checkpoint, so 1B+ models open fast. Weight histograms read one tensor at a
time, with a size cap. Without `config.json` the head count is unknown.
Then the graph shows no heads. For the full graph, run `nnviz serve` and
type the model folder path in the page. Or run `nnviz load <model folder>`.

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

### v0.8 — Scale and fluidity

- [ ] One InstancedMesh per model. Today each layer and head is its own
      mesh: a 27B model costs more than 2 000 draw calls
- [ ] Raycast against a spatial index. Today every click tests every mesh
- [ ] Smooth controls. Rebuild on a throttle, so sliders and fold actions
      never drop frames
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
pip install onnx            # only if you use ONNX files
pip install pillow          # only for image input in live mode
pip install -e .
nnviz model.pth             # writes model.html. Open it in a browser.
nnviz llama_folder/         # HuggingFace folder: config.json + safetensors
nnviz serve                 # drag & drop server at localhost:8000
nnviz live model.pth        # live mode: run the model, watch activations
```

`nnviz` auto-detects the format. It accepts a pickled `nn.Module`, a
`state_dict`, an ONNX file, or a HuggingFace folder. HuggingFace parsing
needs no torch. The output is one self-contained HTML file.
No server and no install needed to view it.

## License

MIT.
