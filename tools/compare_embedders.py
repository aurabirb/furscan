#!/usr/bin/env python3
"""Compare identification results across datasets with different embedders.

Loads one dataset at a time to avoid GPU OOM on low-VRAM machines.

Usage:
    # Single image, no ground truth
    python tools/compare_embedders.py --image path/to/img.jpg

    # Single image with known character
    python tools/compare_embedders.py --image path/to/img.jpg --character "eon_(gryphon)"

    # Directory of test images (character inferred from parent dir name)
    python tools/compare_embedders.py --image-dir path/to/test_images/

    # Specify output file
    python tools/compare_embedders.py --image img.jpg -o results.txt

    # Only compare specific datasets
    python tools/compare_embedders.py --image img.jpg --datasets dino,furtrack
"""

import argparse
import gc
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from sam3_pursuit.api.identifier import (
    FursuitIdentifier,
    SegmentResults,
    discover_datasets,
    detect_embedder,
    merge_multi_dataset_results,
)
from sam3_pursuit.config import Config
from sam3_pursuit.storage.database import Database
from sam3_pursuit.storage.vector_index import VectorIndex


def parse_args():
    parser = argparse.ArgumentParser(description="Compare embedder performance across datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single test image")
    group.add_argument("--image-dir", type=str, help="Directory of test images")
    parser.add_argument("--character", "-c", type=str, help="Ground truth character name (for single image)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results per dataset (default: 10)")
    parser.add_argument("--output", "-o", type=str, default="tools/embedder_comparison_results.txt",
                        help="Output file path")
    parser.add_argument("--datasets", type=str, help="Comma-separated list of dataset names to compare (default: all)")
    parser.add_argument("--device", type=str, help="Device override (cuda, cpu, mps)")
    return parser.parse_args()


def dataset_name(db_path: str) -> str:
    """Extract dataset name from db path: /path/to/foo.db -> foo"""
    return Path(db_path).stem


def load_test_images(args) -> list[tuple[str, str | None]]:
    """Return list of (image_path, ground_truth_character) tuples."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = []

    if args.image:
        images.append((args.image, args.character))
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        if not img_dir.is_dir():
            print(f"Error: {args.image_dir} is not a directory")
            sys.exit(1)
        for f in sorted(img_dir.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                # Infer character from parent directory name
                gt = f.parent.name if f.parent != img_dir else None
                images.append((str(f), gt))
        if not images:
            print(f"Error: no images found in {args.image_dir}")
            sys.exit(1)
    return images


def get_dataset_list(args) -> list[tuple[str, str, str]]:
    """Return list of (name, db_path, index_path) for datasets to compare."""
    all_datasets = discover_datasets()
    if not all_datasets:
        print("Error: no datasets found")
        sys.exit(1)

    filter_names = None
    if args.datasets:
        filter_names = set(args.datasets.split(","))

    result = []
    for db_path, index_path in all_datasets:
        name = dataset_name(db_path)
        if filter_names and name not in filter_names:
            continue
        result.append((name, db_path, index_path))

    if not result:
        print("Error: no matching datasets found")
        sys.exit(1)

    return result


def get_dataset_overview(datasets: list[tuple[str, str, str]]) -> tuple[str, list[dict]]:
    """Format dataset overview table. Lightweight: only reads DB + index, no model loading."""
    lines = ["DATASET OVERVIEW", "=" * 70]
    header = f"{'Dataset':<15} {'Embedder':<12} {'Entries':>10} {'Characters':>12} {'Avg/char':>10}"
    lines.append(header)
    lines.append("-" * 70)

    ds_infos = []
    for name, db_path, index_path in datasets:
        db = Database(db_path)
        stats = db.get_stats()
        emb_name = detect_embedder(db_path, default="unknown")
        try:
            idx = VectorIndex(index_path, embedding_dim=768)
            index_size = idx.size
        except Exception:
            index_size = 0
        total = stats.get("total_detections", 0)
        chars = stats.get("unique_characters", 0)
        avg = total / chars if chars else 0
        lines.append(f"{name:<15} {emb_name:<12} {total:>10,} {chars:>12,} {avg:>10.1f}")
        ds_infos.append({"name": name, "embedder": emb_name, "total": total, "chars": chars})

    lines.append("")
    return "\n".join(lines), ds_infos


def format_segment_results(
    name: str,
    emb_name: str,
    total_entries: int,
    segment_results: list[SegmentResults],
    ground_truth: str | None,
    top_k: int,
) -> tuple[str, dict, str | None]:
    """Format results for one dataset + one image. Returns (text, metrics_dict, top1_char)."""
    lines = []
    metrics = {
        "dataset": name,
        "embedder": emb_name,
        "segments": len(segment_results),
        "top1_correct": False,
        "topk_correct": False,
        "correct_rank": None,
        "top1_confidence": 0.0,
        "top2_confidence": 0.0,
        "confidence_gap": 0.0,
    }
    top1_char = None

    lines.append(f"  Dataset: {name} ({emb_name}, {total_entries:,} entries)")

    if not segment_results:
        lines.append("    (no segments found)")
        return "\n".join(lines), metrics, top1_char

    for seg in segment_results:
        if len(segment_results) > 1:
            lines.append(f"    Segment {seg.segment_index} (bbox={seg.segment_bbox}, conf={seg.segment_confidence:.2f}):")
        if not seg.matches:
            lines.append("    (no matches)")
            continue

        for rank, match in enumerate(seg.matches[:top_k], 1):
            marker = ""
            if ground_truth and match.character_name and match.character_name.lower() == ground_truth.lower():
                marker = " <-- CORRECT"
            lines.append(f"    #{rank}: {match.character_name} ({match.confidence:.1%}){marker}")

        # Metrics from first segment (primary)
        if seg.segment_index == 0 and seg.matches:
            top1 = seg.matches[0]
            top1_char = top1.character_name
            metrics["top1_confidence"] = top1.confidence
            if len(seg.matches) > 1:
                metrics["top2_confidence"] = seg.matches[1].confidence
                metrics["confidence_gap"] = top1.confidence - seg.matches[1].confidence

            if ground_truth:
                for rank, match in enumerate(seg.matches[:top_k], 1):
                    if match.character_name and match.character_name.lower() == ground_truth.lower():
                        if rank == 1:
                            metrics["top1_correct"] = True
                        metrics["topk_correct"] = True
                        metrics["correct_rank"] = rank
                        break

    return "\n".join(lines), metrics, top1_char


def format_summary(
    all_metrics: list[list[dict]],
    top1_chars: list[dict[str, str | None]],
    test_images: list[tuple[str, str | None]],
    ds_names: list[str],
    top_k: int,
) -> str:
    """Format summary statistics across all images."""
    lines = ["SUMMARY", "=" * 70]

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for image_metrics in all_metrics:
        for m in image_metrics:
            by_dataset[m["dataset"]].append(m)

    has_gt = any(gt for _, gt in test_images)
    n_images = len(all_metrics)

    if has_gt:
        lines.append("")
        lines.append(f"Accuracy (over {n_images} image(s), top-k={top_k}):")
        lines.append(f"{'Dataset':<15} {'Embedder':<12} {'Top-1 Acc':>10} {'Top-k Acc':>10} {'Avg Conf':>10} {'Avg Gap':>10}")
        lines.append("-" * 70)

        for ds_name in ds_names:
            metrics = by_dataset.get(ds_name, [])
            if not metrics:
                continue
            emb = metrics[0]["embedder"]
            n = len(metrics)
            top1_acc = sum(1 for m in metrics if m["top1_correct"]) / n * 100
            topk_acc = sum(1 for m in metrics if m["topk_correct"]) / n * 100
            avg_conf = sum(m["top1_confidence"] for m in metrics) / n
            avg_gap = sum(m["confidence_gap"] for m in metrics) / n
            lines.append(f"{ds_name:<15} {emb:<12} {top1_acc:>9.1f}% {topk_acc:>9.1f}% {avg_conf:>9.1%} {avg_gap:>9.1%}")

        # Head-to-head
        if len(ds_names) >= 2:
            lines.append("")
            lines.append("Head-to-head (first segment, top-1):")
            for i in range(len(ds_names)):
                for j in range(i + 1, len(ds_names)):
                    a_name, b_name = ds_names[i], ds_names[j]
                    a_wins, b_wins, ties, both_wrong = 0, 0, 0, 0
                    for image_metrics in all_metrics:
                        a_m = next((m for m in image_metrics if m["dataset"] == a_name), None)
                        b_m = next((m for m in image_metrics if m["dataset"] == b_name), None)
                        if not a_m or not b_m:
                            continue
                        a_ok = a_m["top1_correct"]
                        b_ok = b_m["top1_correct"]
                        if a_ok and b_ok:
                            ties += 1
                        elif a_ok:
                            a_wins += 1
                        elif b_ok:
                            b_wins += 1
                        else:
                            both_wrong += 1
                    total = a_wins + b_wins + ties + both_wrong
                    if total:
                        lines.append(f"  {a_name} vs {b_name}: "
                                     f"{a_name} wins {a_wins}, {b_name} wins {b_wins}, "
                                     f"both correct {ties}, both wrong {both_wrong} (n={total})")
    else:
        lines.append("")
        lines.append("No ground truth provided. Confidence statistics only:")
        lines.append(f"{'Dataset':<15} {'Embedder':<12} {'Avg Top-1 Conf':>15} {'Avg Conf Gap':>15}")
        lines.append("-" * 60)

        for ds_name in ds_names:
            metrics = by_dataset.get(ds_name, [])
            if not metrics:
                continue
            emb = metrics[0]["embedder"]
            n = len(metrics)
            avg_conf = sum(m["top1_confidence"] for m in metrics) / n
            avg_gap = sum(m["confidence_gap"] for m in metrics) / n
            lines.append(f"{ds_name:<15} {emb:<12} {avg_conf:>14.1%} {avg_gap:>14.1%}")

    # Agreement analysis using tracked top-1 characters
    if len(ds_names) >= 2 and top1_chars:
        lines.append("")
        lines.append("Top-1 Agreement:")
        lines.append("-" * 40)
        for i in range(len(ds_names)):
            for j in range(i + 1, len(ds_names)):
                a, b = ds_names[i], ds_names[j]
                agree = sum(
                    1 for tc in top1_chars
                    if tc.get(a) and tc.get(b) and tc[a].lower() == tc[b].lower()
                )
                disagree = sum(
                    1 for tc in top1_chars
                    if tc.get(a) and tc.get(b) and tc[a].lower() != tc[b].lower()
                )
                total = agree + disagree
                pct = agree / total * 100 if total else 0
                lines.append(f"  {a} vs {b}: {agree}/{total} agree ({pct:.1f}%), {disagree} disagree")

                if disagree > 0:
                    lines.append(f"    Disagreements:")
                    for k, tc in enumerate(top1_chars):
                        if tc.get(a) and tc.get(b) and tc[a].lower() != tc[b].lower():
                            img_name = Path(test_images[k][0]).name
                            gt = test_images[k][1] or "?"
                            lines.append(
                                f"      {img_name} (gt={gt}): {a}={tc[a]}, {b}={tc[b]}"
                            )

    lines.append("")
    return "\n".join(lines)


def free_gpu_memory():
    """Release GPU memory between dataset loads."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    args = parse_args()

    test_images = load_test_images(args)
    print(f"\nLoaded {len(test_images)} test image(s)")

    datasets = get_dataset_list(args)
    ds_names = [name for name, _, _ in datasets]
    print(f"Found {len(datasets)} dataset(s): {', '.join(ds_names)}\n")

    output_lines = []
    output_lines.append("EMBEDDER COMPARISON ANALYSIS")
    output_lines.append("=" * 70)
    output_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Test images: {len(test_images)}")
    output_lines.append(f"Datasets: {', '.join(ds_names)}")
    output_lines.append("")

    # Dataset overview (lightweight, no model loading)
    overview_text, ds_infos = get_dataset_overview(datasets)
    output_lines.append(overview_text)

    # Pre-load all test images
    loaded_images: list[tuple[Image.Image | None, str, str | None]] = []
    for img_path, ground_truth in test_images:
        try:
            image = Image.open(img_path).convert("RGB")
            loaded_images.append((image, img_path, ground_truth))
        except Exception as e:
            print(f"  WARNING: Could not load {img_path}: {e}")
            loaded_images.append((None, img_path, ground_truth))

    # Results storage: per-image, per-dataset
    # image_results[img_idx][ds_name] = (text, metrics, top1_char)
    image_results: list[dict[str, tuple[str, dict, str | None]]] = [
        {} for _ in loaded_images
    ]
    # Raw segment results for RRF merge: image_segments[img_idx][ds_name] = list[SegmentResults]
    image_segments: list[dict[str, list[SegmentResults]]] = [
        {} for _ in loaded_images
    ]

    # Process one dataset at a time to avoid GPU OOM
    for ds_idx, (name, db_path, index_path) in enumerate(datasets):
        emb_name = ds_infos[ds_idx]["embedder"]
        total_entries = ds_infos[ds_idx]["total"]

        print(f"\n--- Loading dataset: {name} ({emb_name}) ---")
        try:
            ident = FursuitIdentifier(
                db_path=db_path,
                index_path=index_path,
                device=args.device,
                segmentor_model_name=Config.SAM3_MODEL,
                segmentor_concept=Config.DEFAULT_CONCEPT,
            )
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            continue

        for img_idx, (image, img_path, ground_truth) in enumerate(loaded_images):
            if image is None:
                continue
            print(f"  [{img_idx + 1}/{len(loaded_images)}] {Path(img_path).name} -> {name}")
            try:
                segment_results = ident.identify(image, top_k=args.top_k)
            except Exception as e:
                text = f"  Dataset: {name} ({emb_name}, {total_entries:,} entries)\n    ERROR: {e}"
                image_results[img_idx][name] = (text, {"dataset": name, "embedder": emb_name}, None)
                continue

            text, metrics, top1_char = format_segment_results(
                name, emb_name, total_entries, segment_results, ground_truth, args.top_k
            )
            image_results[img_idx][name] = (text, metrics, top1_char)
            image_segments[img_idx][name] = segment_results

        # Free GPU memory before loading next dataset
        del ident
        free_gpu_memory()
        print(f"--- Done with {name}, freed GPU memory ---")

    # Assemble per-image output
    output_lines.append("PER-IMAGE RESULTS")
    output_lines.append("=" * 70)

    all_metrics: list[list[dict]] = []
    top1_chars: list[dict[str, str | None]] = []

    for img_idx, (image, img_path, ground_truth) in enumerate(loaded_images):
        output_lines.append(f"\nImage: {Path(img_path).name}")
        output_lines.append(f"Path: {img_path}")
        if ground_truth:
            output_lines.append(f"Ground truth: {ground_truth}")
        output_lines.append("")

        if image is None:
            output_lines.append("  ERROR: Could not load image")
            output_lines.append("")
            continue

        img_metrics = []
        img_top1 = {}

        for name in ds_names:
            if name not in image_results[img_idx]:
                continue
            text, metrics, top1_char = image_results[img_idx][name]
            output_lines.append(text)
            output_lines.append("")
            img_metrics.append(metrics)
            img_top1[name] = top1_char

        all_metrics.append(img_metrics)
        top1_chars.append(img_top1)

    # Combined (RRF) results across datasets
    if len(ds_names) >= 2:
        output_lines.append("")
        output_lines.append("COMBINED (Reciprocal Rank Fusion)")
        output_lines.append("=" * 70)

        for img_idx, (image, img_path, ground_truth) in enumerate(loaded_images):
            if image is None:
                continue
            segments_by_ds = image_segments[img_idx]
            if len(segments_by_ds) < 2:
                continue

            all_seg_results = [segments_by_ds[name] for name in ds_names if name in segments_by_ds]
            merged = merge_multi_dataset_results(all_seg_results, top_k=args.top_k)

            output_lines.append(f"\nImage: {Path(img_path).name}")
            if ground_truth:
                output_lines.append(f"Ground truth: {ground_truth}")
            output_lines.append(f"  Combined ({' + '.join(name for name in ds_names if name in segments_by_ds)}):")

            for seg in merged:
                if len(merged) > 1:
                    output_lines.append(f"    Segment {seg.segment_index}:")
                for rank, m in enumerate(seg.matches[:args.top_k], 1):
                    marker = ""
                    if ground_truth and m.character_name and m.character_name.lower() == ground_truth.lower():
                        marker = " <-- CORRECT"
                    output_lines.append(f"    #{rank}: {m.character_name} ({m.confidence:.1%}){marker}")
            output_lines.append("")

    # Summary
    output_lines.append("")
    output_lines.append(format_summary(all_metrics, top1_chars, test_images, ds_names, args.top_k))

    # Write output
    output_path = Config.get_absolute_path(args.output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nResults written to: {output_path}")

    # Also print to stdout
    print("\n" + "=" * 70)
    print("\n".join(output_lines))


if __name__ == "__main__":
    main()
