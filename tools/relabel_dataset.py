#!/usr/bin/env python3
"""Relabel a dataset by re-identifying every detection using avg_embedding.

Creates a copy of the database with character_name updated to the top-1
identification result. The FAISS index is read-only (shared, not copied).

Usage:
    # Dry run (default) - prints proposed changes
    python tools/relabel_dataset.py

    # Actually write changes
    python tools/relabel_dataset.py --apply

    # Custom dataset
    python tools/relabel_dataset.py --dataset pursuit --output pursuit_relabeled
"""

import argparse
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from sam3_pursuit.config import Config
from sam3_pursuit.storage.database import Database
from sam3_pursuit.storage.vector_index import VectorIndex


def relabel(db_path: str, index_path: str, output_db_path: str, top_k: int, apply: bool):
    # Load index (read-only, shared)
    db = Database(db_path)
    index = VectorIndex(index_path, embedding_dim=Config.EMBEDDING_DIM)

    total = index.size
    if total == 0:
        print("Index is empty, nothing to relabel.")
        return

    # Load all detections
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, embedding_id, character_name FROM detections ORDER BY id"
    ).fetchall()
    conn.close()
    print(f"Loaded {len(rows)} detections from {db_path}")

    # Build embedding_id -> character_name lookup for exclusion
    emb_id_to_char: dict[int, str] = {}
    for row_id, emb_id, char_name in rows:
        emb_id_to_char[emb_id] = (char_name or "").lower()

    # For each detection, find top-1 match (excluding self)
    fetch_k = min(top_k * 6 + 1, total)  # +1 for self
    updates: list[tuple[str, int]] = []  # (new_name, detection_id)
    changed = 0
    unchanged = 0
    no_match = 0
    change_counts: Counter = Counter()

    for i, (row_id, emb_id, old_name) in enumerate(rows):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(rows)}...")

        query = index.reconstruct(emb_id).reshape(1, -1).astype(np.float32)
        distances, indices = index.search(query, fetch_k)

        # Group by character, excluding self embedding
        char_indices: dict[str, list[int]] = {}
        char_name_cased: dict[str, str] = {}
        for idx in indices[0]:
            if idx == -1 or idx == emb_id:
                continue
            char = emb_id_to_char.get(int(idx))
            if char is None:
                continue
            char_indices.setdefault(char, []).append(int(idx))
            # Keep first cased version we see
            if char not in char_name_cased:
                det = db.get_detection_by_embedding_id(int(idx))
                if det:
                    char_name_cased[char] = det.character_name or ""

        if not char_indices:
            no_match += 1
            continue

        # Average embeddings per character, pick closest
        best_char = None
        best_dist = float("inf")
        for key, faiss_ids in char_indices.items():
            embs = np.stack([index.reconstruct(i) for i in faiss_ids])
            avg = embs.mean(axis=0, keepdims=True).astype(np.float32)
            dist = float(np.sum((query - avg) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_char = key

        new_name = char_name_cased.get(best_char, best_char)
        old_lower = (old_name or "").lower()

        if old_lower != best_char:
            changed += 1
            change_counts[(old_name or "(none)", new_name)] += 1
            updates.append((new_name, row_id))
        else:
            unchanged += 1

    print(f"\nResults: {changed} changed, {unchanged} unchanged, {no_match} no match")

    if change_counts:
        print(f"\nTop relabeling changes:")
        for (old, new), count in change_counts.most_common(30):
            print(f"  {old} -> {new}  ({count}x)")

    if not apply:
        print(f"\nDry run complete. Use --apply to write {output_db_path}")
        return

    # Copy DB and apply updates
    shutil.copy2(db_path, output_db_path)
    out_conn = sqlite3.connect(output_db_path)
    out_conn.executemany(
        "UPDATE detections SET character_name = ? WHERE id = ?", updates
    )
    out_conn.commit()
    out_conn.close()
    print(f"\nWrote relabeled database to {output_db_path}")


def main():
    parser = argparse.ArgumentParser(description="Relabel dataset using avg_embedding identification")
    parser.add_argument("--dataset", "-d", default="pursuit", help="Source dataset name")
    parser.add_argument("--output", "-o", default=None, help="Output dataset name (default: {dataset}_relabeled)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k neighbors per character for averaging")
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default: dry run)")
    args = parser.parse_args()

    base = Path(Config.BASE_DIR)
    db_path = str(base / f"{args.dataset}.db")
    index_path = str(base / f"{args.dataset}.index")
    output_name = args.output or f"{args.dataset}_relabeled"
    output_db_path = str(base / f"{output_name}.db")

    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)
    if not Path(index_path).exists():
        print(f"Index not found: {index_path}")
        sys.exit(1)

    relabel(db_path, index_path, output_db_path, args.top_k, args.apply)


if __name__ == "__main__":
    main()
