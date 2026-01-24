# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pursuit is a fursuit character recognition system that identifies fursuit characters from photos. It downloads images from FurTrack.com, generates ML embeddings, and provides character identification via CLI and Telegram bot.

**Current System: SAM3 + DINOv2** (`sam3_pursuit/`)
- Uses Meta's Segment Anything Model 3 (SAM3) via ultralytics for fursuit detection/segmentation
- Uses DINOv2 for 768-dimensional embedding generation (better for fine-grained similarity than CLIP)
- FAISS HNSW index for fast approximate nearest neighbor search

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Download images and metadata from FurTrack
python download.py

# CLI usage
python -m sam3_pursuit.api.cli identify photo.jpg           # Identify character
python -m sam3_pursuit.api.cli identify photo.jpg --segment # With segmentation
python -m sam3_pursuit.api.cli add -c "CharName" img1.jpg   # Add character images
python -m sam3_pursuit.api.cli index                        # Index unprocessed images
python -m sam3_pursuit.api.cli stats                        # Show statistics

# Telegram bot (requires TG_BOT_TOKEN in .env)
python tgbot.py

# Run tests
python -m pytest tests/
```

## Architecture

```
download.py                      FurTrack API scraper, SQLite storage, image downloader
        ↓
sam3_pursuit/
├── models/
│   ├── segmentor.py             SAM3 wrapper for fursuit detection
│   └── embedder.py              DINOv2 embedding generator (768D)
├── storage/
│   ├── database.py              SQLite operations for detections
│   └── vector_index.py          FAISS HNSW index
├── pipeline/
│   └── processor.py             Combined segmentation + embedding pipeline
└── api/
    ├── identifier.py            Main public API (SAM3FursuitIdentifier)
    └── cli.py                   Command-line interface
        ↓
tgbot.py                         Telegram bot interface
```

**Storage:**
- SQLite database: `furtrack_sam3.db` (detection metadata)
- FAISS index: `faiss_sam3.index` (768D DINOv2 embeddings)
- Images: `furtrack_images/` directory
- Legacy database: `furtrack.db` (from download.py)

## Key Classes and Functions

**SAM3FursuitIdentifier** (`sam3_pursuit/api/identifier.py`):
- `identify(image, top_k, use_segmentation)` - Find matching characters
- `add_images(character_name, image_paths)` - Add images for a character
- `process_unindexed()` - Batch process images from download database
- `get_stats()` - Get system statistics

**FursuitSegmentor** (`sam3_pursuit/models/segmentor.py`):
- `segment(image)` - Detect and segment fursuit instances

**FursuitEmbedder** (`sam3_pursuit/models/embedder.py`):
- `embed(image)` - Generate L2-normalized DINOv2 embedding
- `embed_batch(images)` - Batch embedding generation

**Download system** (`download.py`):
- `download_all_characters()` - Main download loop
- `get_furtrack_posts()` - Fetch metadata from FurTrack API
- `store_furtrack_data()` / `recall_furtrack_data_by_id()` - SQLite operations

## Environment Variables

- `TG_BOT_TOKEN` - Telegram bot token (required for tgbot.py)

## Device Selection

Code auto-selects: CUDA > MPS (Apple Silicon) > CPU. MacOS requires special env vars (set automatically in config.py):
- `KMP_DUPLICATE_LIB_OK=True`
- `FAISS_OPT_LEVEL=''`

## Database Schema

```sql
-- furtrack_sam3.db: Character detections
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL,
    character_name TEXT,
    embedding_id INTEGER UNIQUE NOT NULL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    confidence REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Model Choices

- **SAM3**: Best-in-class open-vocabulary segmentation for detecting fursuits in images
- **DINOv2**: Self-supervised visual embeddings, superior to CLIP for fine-grained same-object matching
- **FAISS HNSW**: Fast approximate search with configurable accuracy/speed tradeoff
