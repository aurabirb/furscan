#!/usr/bin/env python3
"""Fix Barq placeholder names ("Liked Only") in folders and database.

Resolves placeholder names using profile data from barq_cache.db,
then renames folders in the barq images directory and updates records in the database
(matched by post_id / image UUID).

Usage:
    python tools/fix_barq_names.py              # Dry run (show what would change)
    python tools/fix_barq_names.py --apply      # Apply changes
    python tools/fix_barq_names.py --apply --dataset validation  # Fix a specific dataset
"""

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from sam3_pursuit.config import Config

_PLACEHOLDER_NAMES = {"likes only", "liked only", "private", "mutuals only"}

IMAGES_DIR = f"datasets/{Config.DEFAULT_DATASET}/barq"
DEFAULT_DATASET = Config.DEFAULT_DATASET


def _is_placeholder_name(name: str) -> bool:
    return name.lower().strip() in _PLACEHOLDER_NAMES


def _is_private_social(social: dict) -> bool:
    val = (social.get("value") or "").lower().strip()
    if val in ("private", "@private", "likes only", "@likes only", "liked only", "@liked only"):
        return True
    if val.endswith("mutuals only") or val.endswith("likes only") or val.endswith("liked only"):
        return True
    return False


def get_folder_name_old(profile: dict) -> str:
    """Original name resolution logic (for comparison)."""
    pid = profile.get("id") or profile.get("uuid")
    name = profile.get("username")

    if not name:
        socials = profile.get("socialAccounts") or []
        valid = [s for s in socials if not (
            (s.get("value") or "").lower() in ("private", "@private")
            or (s.get("value") or "").lower().endswith("mutuals only")
        )]
        priority = {"twitter": 0, "telegram": 1, "furAffinity": 2}
        valid.sort(key=lambda s: priority.get(s.get("socialNetwork"), 99))
        if valid:
            name = valid[0].get("value", "").lstrip("@")

    if not name:
        name = profile.get("displayName") or profile.get("uuid", "unknown")

    name = name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("\0", "").replace("|", "_")
    return f"{pid}.{name}"


def get_folder_name_new(profile: dict) -> str:
    """Updated name resolution logic."""
    pid = profile.get("id") or profile.get("uuid")
    name = profile.get("username")
    if name and _is_placeholder_name(name):
        name = None

    if not name:
        socials = profile.get("socialAccounts") or []
        valid = [s for s in socials if not _is_private_social(s)]
        priority = {"twitter": 0, "telegram": 1, "furAffinity": 2}
        valid.sort(key=lambda s: priority.get(s.get("socialNetwork"), 99))
        if valid:
            val = valid[0].get("value", "").lstrip("@")
            if val and not _is_placeholder_name(val):
                name = val

    if not name:
        display = profile.get("displayName")
        if display and not _is_placeholder_name(display):
            name = display
        else:
            name = profile.get("uuid", "unknown")

    name = name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("\0", "").replace("|", "_")
    return f"{pid}.{name}"


def normalize_name(name: str) -> str:
    """Normalize a name the same way the identifier does for database storage."""
    return name.lower().replace(" ", "_")


def get_diffs(cache_db_path: str) -> list[tuple[str, str, str]]:
    """Compare old vs new name resolution, return list of (profile_id, old_label, new_label)."""
    conn = sqlite3.connect(cache_db_path)
    rows = conn.execute("SELECT id, data FROM profiles").fetchall()
    conn.close()

    diffs = []
    for pid, data_json in rows:
        profile = json.loads(data_json)
        old_name = get_folder_name_old(profile)
        new_name = get_folder_name_new(profile)

        if old_name != new_name:
            old_label = old_name.split(".", 1)[1]
            new_label = new_name.split(".", 1)[1]
            diffs.append((pid, old_label, new_label))

    return diffs


def main():
    parser = argparse.ArgumentParser(description="Fix Barq placeholder names")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--dataset", "-ds", default=DEFAULT_DATASET, help="Dataset name (default: pursuit)")
    parser.add_argument("--images-dir", default=IMAGES_DIR, help="Barq images directory (default: barq_images)")
    parser.add_argument("--cache-db", default="barq_cache.db", help="Barq cache database (default: barq_cache.db)")
    args = parser.parse_args()

    db_path = f"{args.dataset}.db"
    cache_db_path = args.cache_db
    images_dir = Path(args.images_dir)

    # Validate paths
    if not Path(cache_db_path).exists():
        print(f"Error: Cache database not found: {cache_db_path}")
        return 1

    # Get diffs
    print(f"Comparing old vs new name resolution from {cache_db_path}...")
    diffs = get_diffs(cache_db_path)

    if not diffs:
        print("No changes needed.")
        return 0

    # Filter to only placeholder fixes
    placeholder_diffs = [(pid, old, new) for pid, old, new in diffs if old.lower().strip() in _PLACEHOLDER_NAMES]
    other_diffs = [(pid, old, new) for pid, old, new in diffs if old.lower().strip() not in _PLACEHOLDER_NAMES]

    print(f"\nTotal profiles changed: {len(diffs)}")
    print(f"  Placeholder names to fix: {len(placeholder_diffs)}")
    if other_diffs:
        print(f"  Other changes (skipped): {len(other_diffs)}")
    print()

    if not placeholder_diffs:
        print("No placeholder names to fix.")
        return 0

    # Show diff
    print("=== Placeholder name resolutions ===")
    for pid, old, new in sorted(placeholder_diffs, key=lambda x: x[2].lower()):
        print(f"  {pid}: '{old}' -> '{new}'")
    print()

    # Phase 1: Scan folders on disk
    print(f"--- Folder renames ({images_dir}/) ---")
    folder_renames = []
    folder_merges = []
    folder_missing = []
    # Map: profile_id -> (old_folder, new_label, list of image UUIDs in old folder)
    profile_image_uuids: dict[str, list[str]] = {}

    for pid, old_label, new_label in placeholder_diffs:
        old_folder = images_dir / f"{pid}.{old_label}"
        new_folder = images_dir / f"{pid}.{new_label}"

        if not old_folder.exists():
            folder_missing.append((pid, old_label, new_label))
            continue

        old_images = list(old_folder.glob("*.jpg")) + list(old_folder.glob("*.png")) + list(old_folder.glob("*.jpeg"))
        image_uuids = [img.stem for img in old_images]
        profile_image_uuids[pid] = image_uuids

        if new_folder.exists():
            folder_merges.append((pid, old_label, new_label, len(old_images)))
        else:
            folder_renames.append((pid, old_label, new_label, len(old_images)))

    for pid, old, new, count in folder_renames:
        print(f"  RENAME: {pid}.{old} -> {pid}.{new} ({count} images)")

    for pid, old, new, count in folder_merges:
        print(f"  MERGE:  {pid}.{old} -> {pid}.{new} ({count} images to move)")

    if folder_missing:
        print(f"  ({len(folder_missing)} profiles have no folder on disk)")

    # Phase 2: Plan database updates matched by post_id
    print(f"\n--- Database updates ({db_path}) ---")
    # Each update: (profile_id, new_normalized_name, list of post_ids, merge_into_count)
    db_updates: list[tuple[str, str, list[str], int]] = []

    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)

        for pid, old_label, new_label in placeholder_diffs:
            new_normalized = normalize_name(new_label)
            image_uuids = profile_image_uuids.get(pid, [])

            if not image_uuids:
                continue

            # Find records for this profile's images that have the placeholder name
            placeholders = ",".join("?" for _ in image_uuids)
            matching = conn.execute(
                f"SELECT COUNT(*) FROM detections WHERE source='barq' AND character_name='liked_only' AND post_id IN ({placeholders})",
                image_uuids
            ).fetchone()[0]

            if matching == 0:
                continue

            # Check if target name already exists
            existing = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE character_name=?",
                (new_normalized,)
            ).fetchone()[0]

            db_updates.append((pid, new_normalized, image_uuids, matching, existing))

        conn.close()

        for pid, new_norm, uuids, count, existing in db_updates:
            if existing > 0:
                print(f"  MERGE: liked_only -> '{new_norm}' ({count} records, merging into {existing} existing) [profile {pid}]")
            else:
                print(f"  RENAME: liked_only -> '{new_norm}' ({count} records) [profile {pid}]")

        total_db_records = sum(count for _, _, _, count, _ in db_updates)
        print(f"  Total: {total_db_records} records across {len(db_updates)} profiles")

        if not db_updates:
            print("  No matching records in database")
    else:
        print(f"  Database not found: {db_path}")

    # Apply changes
    if not args.apply:
        print(f"\nDry run complete. Use --apply to execute these changes.")
        return 0

    print(f"\n{'='*50}")
    print("APPLYING CHANGES")
    print(f"{'='*50}\n")

    # Apply folder renames
    for pid, old, new, count in folder_renames:
        old_folder = images_dir / f"{pid}.{old}"
        new_folder = images_dir / f"{pid}.{new}"
        old_folder.rename(new_folder)
        print(f"  Renamed: {pid}.{old} -> {pid}.{new}")

    # Apply folder merges (move files into existing folder, then remove old)
    for pid, old, new, count in folder_merges:
        old_folder = images_dir / f"{pid}.{old}"
        new_folder = images_dir / f"{pid}.{new}"
        moved = 0
        for img in list(old_folder.glob("*")):
            dest = new_folder / img.name
            if not dest.exists():
                shutil.move(str(img), str(dest))
                moved += 1
            else:
                img.unlink()  # Duplicate, remove
        if not any(old_folder.iterdir()):
            old_folder.rmdir()
        print(f"  Merged: {pid}.{old} -> {pid}.{new} ({moved} files moved)")

    # Apply database updates (by post_id)
    if db_updates and Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        total_updated = 0
        for pid, new_norm, uuids, count, existing in db_updates:
            placeholders = ",".join("?" for _ in uuids)
            conn.execute(
                f"UPDATE detections SET character_name=? WHERE source='barq' AND character_name='liked_only' AND post_id IN ({placeholders})",
                [new_norm] + uuids
            )
            total_updated += count
            print(f"  Updated DB: liked_only -> '{new_norm}' ({count} records) [profile {pid}]")
        conn.commit()
        conn.close()
        print(f"  Total updated: {total_updated} records")

    # Clean empty folders
    empty_removed = 0
    if images_dir.exists():
        for d in list(images_dir.iterdir()):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                empty_removed += 1

    print(f"\nDone! Applied {len(folder_renames)} renames, {len(folder_merges)} merges, {len(db_updates)} DB updates.")
    if empty_removed:
        print(f"Removed {empty_removed} empty directories.")
    return 0


if __name__ == "__main__":
    exit(main())
