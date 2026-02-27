"""
Dataset Fixer
Automatically fixes the issues found by audit_dataset.py:
1. Removes the cross-class duplicate (keeps the violence copy, removes non_violence copy)
2. Removes the 12 too-short videos from non_violence
3. Reports the 18 same-class duplicates (keeps one, removes extras)

Run this ONCE before training. It moves files to a _removed/ backup folder
instead of deleting them permanently, so you can restore if needed.
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

DATASET_PATH = r'C:\Users\Jense\Documents\CODING-C\ViolenceDetection\ViolenceDetection\TRAINING\training_datasets_V2'
BACKUP_DIR   = r'C:\Users\Jense\Documents\CODING-C\ViolenceDetection\ViolenceDetection\TRAINING\training_datasets_V2\_removed'

MIN_FRAMES       = 8
MIN_DURATION_SEC = 1.0

import cv2
import numpy as np

def get_file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    return frames, frames / fps

def move_to_backup(path, reason):
    backup = Path(BACKUP_DIR) / reason
    backup.mkdir(parents=True, exist_ok=True)
    dest = backup / Path(path).name
    # avoid name collision in backup
    if dest.exists():
        dest = backup / f"{Path(path).stem}__{Path(path).parent.name}{Path(path).suffix}"
    shutil.move(str(path), str(dest))
    print(f"  MOVED [{reason}]: {Path(path).name} → _removed/{reason}/")

def main():
    categories = ['non_violence', 'violence']
    exts = ('.mp4', '.avi', '.mov', '.mkv')

    Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)

    all_hashes = defaultdict(list)   # hash -> [(category, path)]
    all_files  = {cat: [] for cat in categories}

    # --- Scan all files ---
    print("Scanning dataset...")
    for cat in categories:
        folder = Path(DATASET_PATH) / cat
        files = [f for f in folder.iterdir() if f.suffix.lower() in exts]
        all_files[cat] = files
        for f in tqdm(files, desc=f'  Hashing {cat}'):
            h = get_file_hash(str(f))
            all_hashes[h].append((cat, f))

    removed = 0

    # --- Fix 1: Cross-class duplicates ---
    # Keep the violence copy (more valuable label), remove non_violence copy
    print("\n--- Fix 1: Cross-class duplicates ---")
    for file_hash, entries in all_hashes.items():
        cats = [e[0] for e in entries]
        if len(set(cats)) > 1:
            # remove the non_violence copy — it's mislabeled or ambiguous
            for cat, path in entries:
                if cat == 'non_violence':
                    move_to_backup(path, 'cross_class_duplicate')
                    removed += 1

    # --- Fix 2: Too-short videos ---
    print("\n--- Fix 2: Too-short videos (< 1s or < 8 frames) ---")
    for cat in categories:
        folder = Path(DATASET_PATH) / cat
        files = [f for f in folder.iterdir() if f.suffix.lower() in exts]
        for f in files:
            frames, duration = get_video_info(str(f))
            if frames < MIN_FRAMES or duration < MIN_DURATION_SEC:
                move_to_backup(str(f), 'too_short')
                removed += 1

    # --- Fix 3: Same-class duplicates ---
    # Keep the first occurrence, remove extras
    print("\n--- Fix 3: Same-class duplicates ---")
    for file_hash, entries in all_hashes.items():
        cats = [e[0] for e in entries]
        if len(entries) > 1 and len(set(cats)) == 1:
            # all same class — keep first, remove rest
            for cat, path in entries[1:]:
                if path.exists():
                    move_to_backup(str(path), 'same_class_duplicate')
                    removed += 1

    # --- Final count ---
    print("\n" + "=" * 50)
    print(f"Done. {removed} files moved to: {BACKUP_DIR}")
    print("Files are NOT deleted — restore from _removed/ if needed.")
    print("\nFinal dataset size:")
    for cat in categories:
        folder = Path(DATASET_PATH) / cat
        count = len([f for f in folder.iterdir() if f.suffix.lower() in exts])
        print(f"  {cat}: {count} videos")
    print("\nYou can now retrain. Delete the old cache folder first:")
    print(r"  training\caches\pose_cache_lstm_seq20_k2_simple")
    print(r"  training\caches\pose_cache_lstm_seq20_k2_motion")

if __name__ == '__main__':
    main()
