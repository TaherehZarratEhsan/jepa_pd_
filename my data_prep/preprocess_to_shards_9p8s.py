"""
Preprocess distance signals into WebDataset shards (simple version).

- Target duration: 9.8 seconds
- Target FPS: 25 (50 fps -> downsample by taking every other frame)
- Skips samples that are too short after resampling


Output:
  - 4 WebDataset shards with ~1000 samples each
  - Each sample contains: signal.npy (length=245), features.npy, metadata
"""

import hashlib
import os
import pickle
from pathlib import Path
import numpy as np
import re
import webdataset as wds
from typing import Optional, Dict
from feature_extractor import feature_ext_analysis

TARGET_FPS = 25
TARGET_DURATION = 9.8
TARGET_LEN = int(TARGET_FPS * TARGET_DURATION)  # 245

# --------- USER SETTINGS (edit here) ----------
ANNOTATED_PKL = r'/data/diag/Tahereh/new/jepaa/data/video_keypoints.pkl'
# Always write into the existing `my data_prep/` directory, even if you run the script from inside it.
OUTPUT_DIR = str(Path(__file__).resolve().parent)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1  # test = 1 - train - val
TRAIN_SHARDS = 4
VAL_SHARDS = 1
TEST_SHARDS = 1
HASH_SEED = 42  # reserved; md5 deterministic
# ----------------------------------------------


def process_distance_signal(distances, fps, target_fps=TARGET_FPS, target_len=TARGET_LEN):
    # Skip if fps is not 50 or 25
    if fps not in [50, 25]:
        return None
    
    arr = np.asarray(distances, dtype=np.float32)

    # fast path: 50 -> 25 by taking every other sample
    if fps == 50 and target_fps == 25:
        arr = arr[::2]
    elif fps == target_fps:
        pass

    # Truncate to target length if longer
    if len(arr) > target_len:
        arr = arr[:target_len]

    return arr


def choose_split(pid: str, train_ratio: float, val_ratio: float) -> str:
    """Deterministically assign a split based on patient id hash."""
    h = hashlib.md5(str(pid).encode()).hexdigest()
    r = int(h, 16) % 10_000_000 / 10_000_000.0
    if r < train_ratio:
        return "train"
    if r < train_ratio + val_ratio:
        return "val"
    return "test"


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load annotated pickle
    with open(ANNOTATED_PKL, 'rb') as f:
        annotated = pickle.load(f)

    combined_distances = annotated.get('distances', [])
    combined_fps = annotated.get('fps', [])
    combined_ids = annotated.get('id', [])
    combined_paths = annotated.get('video_path', [""] * len(combined_distances))
    combined_labels = annotated.get('label', [""] * len(combined_distances))

 
    total = len(combined_distances)

    # Instantiate feature extractor with config
    cfg = {
        'test_type': 'ft',
    }
    feature_inst = feature_ext_analysis(cfg)

    # Ratios and shard sizes
    test_ratio = max(0.0, 1.0 - TRAIN_RATIO - VAL_RATIO)
    ratios = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": test_ratio}
    shard_counts = {"train": max(1, TRAIN_SHARDS), "val": max(1, VAL_SHARDS), "test": max(1, TEST_SHARDS)}
    est_counts = {
        split: max(1, int(np.ceil((total * ratios[split]) / shard_counts[split]))) if ratios[split] > 0 else 1
        for split in ratios
    }
    writers: Dict[str, wds.ShardWriter] = {}
    for split in ["train", "val", "test"]:
        template = Path(os.path.join(OUTPUT_DIR, f"signals-{split}-%06d.tar")).resolve().as_posix()
        writers[split] = wds.ShardWriter("file:///" + template, maxcount=est_counts[split])

    written = 0
    skipped = 0

    print(f"Total samples to process: {total}")
    print(f"Writing shards to: {OUTPUT_DIR}")
    print(f"Train/Val/Test ratios: {TRAIN_RATIO}/{VAL_RATIO}/{test_ratio}")
    print(f"Shard counts (train/val/test): {shard_counts}")
    print(f"Estimated samples per shard: {est_counts}")
    print(f"Skipping samples shorter than target length ({TARGET_LEN} frames)\n")

    for i in range(total):
            raw = combined_distances[i]
            fps = combined_fps[i]
            sid = combined_ids[i] 
            vpath = combined_paths[i] 
            label = combined_labels[i] 

            # Resample/downsample to 9.8s @ 25fps
            processed = process_distance_signal(raw, fps)
            
            # Skip if fps is not supported (50 or 25)
            if processed is None:
                skipped += 1
                continue

            # Skip if too short
            if len(processed) < TARGET_LEN:
                skipped += 1
                continue

            # Compute features
            features, feat_name = feature_inst._extract_features(processed, TARGET_FPS)
            features_arr = np.asarray(features, dtype=np.float32)

            # Build key from path: extract visit number, On/Off and L/R
            path_str = str(vpath)
            # visit number
            m = re.search(r"[Vv]isit\s*(\d+)", path_str)
            visit = m.group(1) if m else "unknown"
            # On/Off — allow separators (underscore, dash, slash, space) around the token
            onoff_m = re.search(r"(?:^|[_/\\\s-])(On|Off)(?:[_/\\\s-]|$|\.)", path_str, re.IGNORECASE)
            onoff = onoff_m.group(1).lower() if onoff_m else "unknown"
            # L or R (look for L_cropped or R_cropped or pattern like 2L)
            lr = None
            if re.search(r"L_cropped", path_str, re.IGNORECASE):
                lr = "L"
            elif re.search(r"R_cropped", path_str, re.IGNORECASE):
                lr = "R"
            else:
                m2 = re.search(r"(\d+)?([LR])_cropped", path_str, re.IGNORECASE)
                if m2:
                    lr = m2.group(2).upper()
                else:
                    # fallback: look for plain ' L ' or '_L' or 'L.' before extension
                    if re.search(r"[^A-Za-z0-9]L[^A-Za-z0-9]", path_str):
                        lr = "L"
                    elif re.search(r"[^A-Za-z0-9]R[^A-Za-z0-9]", path_str):
                        lr = "R"


            # subject id (sid) sanitized
            sid_str = str(sid).replace(" ", "_")

            key = f"{sid_str}_visit{visit}_{onoff}_{lr}"

            split = choose_split(sid, TRAIN_RATIO, VAL_RATIO)
            writer = writers[split]
            writer.write({
                "signal.npy": processed,
                "features.npy": features_arr,
                "video_path": str(vpath),
                "label": str(label),
                "__key__": key,
            })

            written += 1
            if written % 100 == 0:
                print(f"  Progress: {written} written, {skipped} skipped...")



    for w in writers.values():
        w.close()

    print(f"\n✓ Preprocessing complete!")
    print(f"  Written: {written}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
