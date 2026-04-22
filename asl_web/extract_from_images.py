"""
Extract MediaPipe hand landmarks from ASL_Dataset.
Reads images DIRECTLY from the zip (no full extraction needed).
Produces data/user_dataset.csv  (42 features + label, same as american.csv)

Usage:
    python extract_from_images.py
"""

import os, csv, zipfile
import cv2
import numpy as np
import mediapipe as mp

# ── Config ──────────────────────────────────────────────────────────────────
ZIP_PATH      = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                              "Downloads", "ASL_Dataset.zip")
OUT_CSV       = os.path.join(os.path.dirname(__file__), "data", "user_dataset.csv")
MAX_PER_CLASS = 100          # images per letter  (100 × 26 = 2600 total)
SKIP_LABELS   = {"nothing", "space", "del"}
# ────────────────────────────────────────────────────────────────────────────

_MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
with open(_MODEL_FILE, "rb") as f:
    _buf = f.read()

_opts = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_buffer=_buf),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
)
detector = mp.tasks.vision.HandLandmarker.create_from_options(_opts)


def extract_from_bytes(img_bytes):
    """Run MediaPipe on raw image bytes; return 42-float list or None."""
    arr     = np.frombuffer(img_bytes, dtype=np.uint8)
    img     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.flip(img_rgb, 1)
    result  = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    if not result.hand_landmarks:
        return None
    row = []
    for lm in result.hand_landmarks[0]:
        row.append(lm.x)
        row.append(lm.y)
    return row[:42]


print(f"Reading from: {ZIP_PATH}")
if not os.path.exists(ZIP_PATH):
    print("[ERROR] zip not found. Check ZIP_PATH.")
    raise SystemExit(1)

# Group zip entries by class label
from collections import defaultdict
class_entries = defaultdict(list)
with zipfile.ZipFile(ZIP_PATH) as zf:
    for name in zf.namelist():
        parts = name.replace("\\", "/").split("/")
        # expect  Train/<label>/<file>.jpg
        if len(parts) == 3 and parts[0] == "Train" and parts[2]:
            label = parts[1].upper()
            if label.lower() not in SKIP_LABELS and len(label) == 1:
                class_entries[label].append(name)

total = 0
with zipfile.ZipFile(ZIP_PATH) as zf, open(OUT_CSV, "w", newline="") as csvf:
    writer = csv.writer(csvf)
    for label in sorted(class_entries):
        entries = class_entries[label][:MAX_PER_CLASS * 3]  # read extra to hit quota
        count   = 0
        for entry in entries:
            if count >= MAX_PER_CLASS:
                break
            feats = extract_from_bytes(zf.read(entry))
            if feats:
                writer.writerow(feats + [label])
                count += 1
        print(f"  {label}: {count} samples")
        total += count

print(f"\nDone. {total} rows  →  {OUT_CSV}")
