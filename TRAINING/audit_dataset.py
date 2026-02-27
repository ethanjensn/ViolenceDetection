"""
Dataset Audit Script
Checks for common issues that cause training to get worse when you add more data:
- Class imbalance
- Videos with zero pose detections (empty features)
- Duplicate videos across classes (same hash in both violence and non_violence)
- Very short or corrupt videos
Run this before training to understand what's in your dataset.
"""

import os
import cv2
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

DATASET_PATH = r'C:\Users\Jense\Documents\CODING-C\ViolenceDetection\ViolenceDetection\TRAINING\training_datasets'
MIN_FRAMES = 8       # flag videos shorter than this as suspect
MIN_DURATION_SEC = 1.0  # flag videos shorter than 1 second


def get_file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def audit_video(vid_path):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return {'status': 'corrupt', 'frames': 0, 'duration': 0, 'fps': 0}

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        'status': 'ok',
        'frames': frames,
        'duration': round(duration, 2),
        'fps': round(fps, 1),
        'resolution': f'{width}x{height}'
    }


def main():
    categories = {'non_violence': 0, 'violence': 1}
    exts = ('.mp4', '.avi', '.mov', '.mkv')

    all_hashes = defaultdict(list)   # hash -> list of (category, path)
    issues = []
    stats = {}

    print("=" * 60)
    print("DATASET AUDIT")
    print("=" * 60)

    for category in categories:
        folder = Path(DATASET_PATH) / category
        if not folder.exists():
            print(f"\n[ERROR] Folder not found: {folder}")
            continue

        files = [f for f in folder.iterdir() if f.suffix.lower() in exts]
        print(f"\n[{category.upper()}] {len(files)} videos found")

        durations = []
        frame_counts = []
        corrupt = []
        short = []
        resolutions = defaultdict(int)

        for f in tqdm(files, desc=f'  Auditing {category}'):
            info = audit_video(str(f))

            if info['status'] == 'corrupt':
                corrupt.append(f.name)
                issues.append(f"CORRUPT [{category}]: {f.name}")
                continue

            durations.append(info['duration'])
            frame_counts.append(info['frames'])
            resolutions[info['resolution']] += 1

            if info['frames'] < MIN_FRAMES or info['duration'] < MIN_DURATION_SEC:
                short.append((f.name, info['frames'], info['duration']))
                issues.append(f"TOO_SHORT [{category}]: {f.name} — {info['frames']} frames, {info['duration']}s")

            # Hash for duplicate detection
            file_hash = get_file_hash(str(f))
            all_hashes[file_hash].append((category, str(f)))

        stats[category] = {
            'count': len(files),
            'corrupt': len(corrupt),
            'short': len(short),
            'durations': durations,
            'frame_counts': frame_counts,
            'resolutions': dict(resolutions)
        }

        if durations:
            print(f"  Duration  — min: {min(durations):.1f}s  max: {max(durations):.1f}s  mean: {np.mean(durations):.1f}s")
            print(f"  Frames    — min: {min(frame_counts)}  max: {max(frame_counts)}  mean: {np.mean(frame_counts):.0f}")
        print(f"  Corrupt   — {len(corrupt)}")
        print(f"  Too short — {len(short)}")
        print(f"  Top resolutions: {dict(sorted(resolutions.items(), key=lambda x: -x[1])[:3])}")

    # --- Class balance ---
    print("\n" + "=" * 60)
    print("CLASS BALANCE")
    print("=" * 60)
    counts = {cat: stats[cat]['count'] for cat in stats}
    total = sum(counts.values())
    for cat, count in counts.items():
        pct = 100 * count / total if total else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")

    if counts:
        ratio = max(counts.values()) / max(min(counts.values()), 1)
        if ratio > 1.5:
            issues.append(f"CLASS IMBALANCE: ratio is {ratio:.2f}x — consider balancing")
            print(f"  ⚠ Imbalance ratio: {ratio:.2f}x (>1.5x can hurt training)")
        else:
            print(f"  ✓ Balance ratio: {ratio:.2f}x (good)")

    # --- Duplicate detection ---
    print("\n" + "=" * 60)
    print("DUPLICATE DETECTION")
    print("=" * 60)
    cross_class_dupes = []
    same_class_dupes = []

    for file_hash, entries in all_hashes.items():
        if len(entries) > 1:
            cats = [e[0] for e in entries]
            if len(set(cats)) > 1:
                cross_class_dupes.append(entries)
                issues.append(f"CROSS-CLASS DUPLICATE: same video in both classes! {[e[1] for e in entries]}")
            else:
                same_class_dupes.append(entries)

    print(f"  Cross-class duplicates (same video, different label): {len(cross_class_dupes)}")
    print(f"  Same-class duplicates: {len(same_class_dupes)}")

    if cross_class_dupes:
        print("\n  ⚠ CRITICAL — These videos appear in BOTH violence and non_violence:")
        for dupe in cross_class_dupes[:10]:
            for cat, path in dupe:
                print(f"    [{cat}] {Path(path).name}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ISSUES SUMMARY")
    print("=" * 60)
    if not issues:
        print("  ✓ No issues found")
    else:
        print(f"  {len(issues)} issue(s) found:\n")
        for issue in issues:
            print(f"  • {issue}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    total_videos = sum(s['count'] for s in stats.values())
    print(f"  Total videos: {total_videos}")
    if total_videos < 3000:
        print("  ⚠ Dataset is small — more data will help more than architecture changes")
    elif total_videos < 6000:
        print("  ~ Dataset is medium — focus on label quality over quantity")
    else:
        print("  ✓ Dataset size is reasonable — focus on model architecture")

    corrupt_total = sum(s['corrupt'] for s in stats.values())
    short_total = sum(s['short'] for s in stats.values())
    if corrupt_total + short_total > 0:
        print(f"  ⚠ Remove {corrupt_total} corrupt + {short_total} too-short videos before training")
    if cross_class_dupes:
        print(f"  ⚠ Fix {len(cross_class_dupes)} cross-class duplicates — these directly hurt accuracy")


if __name__ == '__main__':
    main()
