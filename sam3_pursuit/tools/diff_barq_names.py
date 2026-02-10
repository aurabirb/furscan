#!/usr/bin/env python3
"""Compare old vs new Barq name resolution for all cached profiles.

Reads all profiles from barq_cache.db and shows what would change
with the updated get_folder_name logic.
"""

import json
import sqlite3
from pathlib import Path

CACHE_DB = Path(__file__).parent.parent / "barq_cache.db"

# --- Old logic (verbatim from before the fix) ---

def _is_private_social_old(social: dict) -> bool:
    val = (social.get("value") or "").lower()
    return val in ("private", "@private") or val.endswith("mutuals only")


def get_folder_name_old(profile: dict) -> str:
    pid = profile.get("id") or profile.get("uuid")
    name = profile.get("username")

    if not name:
        socials = profile.get("socialAccounts") or []
        valid = [s for s in socials if not _is_private_social_old(s)]
        priority = {"twitter": 0, "telegram": 1, "furAffinity": 2}
        valid.sort(key=lambda s: priority.get(s.get("socialNetwork"), 99))
        if valid:
            name = valid[0].get("value", "").lstrip("@")

    if not name:
        name = profile.get("displayName") or profile.get("uuid", "unknown")

    name = name.replace("/", "_").replace("\\", "_").replace(":", "_").replace("\0", "").replace("|", "_")
    return f"{pid}.{name}"


# --- New logic (matching updated download_barq.py) ---

_PLACEHOLDER_NAMES = {"likes only", "liked only", "private", "mutuals only"}


def _is_placeholder_name(name: str) -> bool:
    return name.lower().strip() in _PLACEHOLDER_NAMES


def _is_private_social_new(social: dict) -> bool:
    val = (social.get("value") or "").lower().strip()
    if val in ("private", "@private", "likes only", "@likes only", "liked only", "@liked only"):
        return True
    if val.endswith("mutuals only") or val.endswith("likes only") or val.endswith("liked only"):
        return True
    return False


def get_folder_name_new(profile: dict) -> str:
    pid = profile.get("id") or profile.get("uuid")
    name = profile.get("username")
    if name and _is_placeholder_name(name):
        name = None

    if not name:
        socials = profile.get("socialAccounts") or []
        valid = [s for s in socials if not _is_private_social_new(s)]
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


def main():
    conn = sqlite3.connect(str(CACHE_DB))
    rows = conn.execute("SELECT id, data FROM profiles").fetchall()
    conn.close()

    diffs = []
    same = 0

    for pid, data_json in rows:
        profile = json.loads(data_json)
        old_name = get_folder_name_old(profile)
        new_name = get_folder_name_new(profile)

        if old_name != new_name:
            old_label = old_name.split(".", 1)[1]
            new_label = new_name.split(".", 1)[1]
            diffs.append((pid, old_label, new_label))
        else:
            same += 1

    print(f"Total profiles: {len(rows)}")
    print(f"Unchanged: {same}")
    print(f"Changed: {len(diffs)}")
    print()

    if diffs:
        placeholder_fixed = []
        other = []

        for pid, old, new in diffs:
            if old.lower().strip() in _PLACEHOLDER_NAMES:
                placeholder_fixed.append((pid, old, new))
            else:
                other.append((pid, old, new))

        if placeholder_fixed:
            print(f"=== Placeholder names fixed ({len(placeholder_fixed)}) ===")
            for pid, old, new in sorted(placeholder_fixed, key=lambda x: x[1].lower()):
                print(f"  {pid}: '{old}' -> '{new}'")
            print()

        if other:
            print(f"=== Other changes ({len(other)}) ===")
            for pid, old, new in sorted(other, key=lambda x: x[1].lower()):
                print(f"  {pid}: '{old}' -> '{new}'")
            print()


if __name__ == "__main__":
    main()
