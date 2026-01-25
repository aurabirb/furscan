# Pursuit - Fursuit Character Recognition

Fursuit character recognition system using computer vision. Identifies fursuit characters from photos by matching against a database of known characters.

## Current Status

**Working:** SAM3 + DINOv2 pipeline on CUDA

The system uses SAM3 text prompts with `"fursuiter"` for automatic detection of all fursuits in an image.

## Installation

### 1. Setup Python environment

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

### 2. Login to HuggingFace (required for SAM3)

```bash
pip install huggingface_hub
huggingface-cli login
```

### 3. Download SAM3 model (~3.5GB)

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='.')"
```

Or manually download from https://huggingface.co/facebook/sam3 and place `sam3.pt` in the project root.

### 4. Verify installation

```bash
python -c "from sam3_pursuit.models.segmentor import FursuitSegmentor; s = FursuitSegmentor(); print('SAM3 ready!')"
```

## Quick Start

```bash
# Identify character in photo
python -m sam3_pursuit.api.cli identify photo.jpg

# Add images for a character
python -m sam3_pursuit.api.cli add -c "CharName" img1.jpg img2.jpg

# View statistics
python -m sam3_pursuit.api.cli stats

# Run Telegram bot
TG_BOT_TOKEN=xxx python tgbot.py

# Run tests
python -m pytest tests/
python scripts/test_segmentation.py segment
python scripts/test_segmentation.py visualize
```

## Architecture

```
Image → SAM3 (segment by "fursuiter") → DINOv2 (embed) → FAISS (search) → Results
```

```
sam3_pursuit/
├── models/
│   ├── segmentor.py      SAM3 segmentation with text prompts
│   └── embedder.py       DINOv2 768D embeddings
├── storage/
│   ├── database.py       SQLite detection metadata
│   └── vector_index.py   FAISS HNSW index
├── pipeline/
│   └── processor.py      Segmentation + embedding pipeline
└── api/
    ├── identifier.py     Main API: SAM3FursuitIdentifier
    └── cli.py            Command-line interface

scripts/
└── test_segmentation.py  Debug and visualization

tgbot.py                  Telegram bot
```

## Key APIs

```python
from sam3_pursuit import SAM3FursuitIdentifier

identifier = SAM3FursuitIdentifier()

# Identify character in image
results = identifier.identify(image, top_k=5)

# Add images for a character
identifier.add_images("CharacterName", ["img1.jpg", "img2.jpg"])

# Segment (default concept: "fursuiter")
segmentor.segment(image)
segmentor.segment(image, concept="person")  # custom concept
```

## Storage

| File | Contents |
|------|----------|
| `*.db` | Detection metadata (SQLite) |
| `*.index` | 768D embeddings (FAISS HNSW) |

## Config

Key settings in `sam3_pursuit/config.py`:

```python
SAM3_MODEL = "sam3"
DINOV2_MODEL = "facebook/dinov2-base"
EMBEDDING_DIM = 768
DEFAULT_CONCEPT = "fursuiter"  # SAM3 text prompt
```

## Environment

- `TG_BOT_TOKEN` - Telegram bot token
- `HF_TOKEN` - HuggingFace token (for SAM3 download)

Device auto-selection: CUDA > MPS > CPU

## References

- [SAM3 Paper](https://arxiv.org/abs/2511.16719) - Segment Anything with Concepts
- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [DINOv2](https://github.com/facebookresearch/dinov2) - Self-supervised vision
