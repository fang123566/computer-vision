"""
Flask backend for real-time ASL recognition.
  GET  /         – frontend
  GET  /health   – server status check
  POST /predict  – base64 JPEG → letter + landmarks

Uses MediaPipe 0.10+ Tasks API (HandLandmarker).
"""

import os, base64
import cv2
import mediapipe as mp
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# MediaPipe hand connection pairs (21 landmarks)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

_MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
with open(_MODEL_FILE, "rb") as _f:
    _model_buf = _f.read()
_base_options = mp.tasks.BaseOptions(model_asset_buffer=_model_buf)
_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=_base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
_detector = mp.tasks.vision.HandLandmarker.create_from_options(_options)


def _hand_reach(lm_list):
    """Wrist-to-middle-fingertip distance – real gesture hand scores highest."""
    wrist = lm_list[0]
    tip   = lm_list[12]  # middle fingertip
    return ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2) ** 0.5


def _hand_center_score(lm_list):
    """Center-priority score in [0,1]. Hands near frame center score higher."""
    cx = sum(lm.x for lm in lm_list) / len(lm_list)
    cy = sum(lm.y for lm in lm_list) / len(lm_list)
    dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    return max(0.0, 1.0 - dist / 0.72)


def _normalize(features):
    """Position- and scale-invariant normalization of 42-element landmark list.
    Translates wrist to origin, scales by wrist-to-middle-MCP distance.
    Must match normalization used in train_model.py.
    """
    wx, wy = features[0], features[1]
    mx, my = features[18], features[19]   # landmark 9 = middle finger MCP
    scale  = max(((mx - wx)**2 + (my - wy)**2) ** 0.5, 1e-6)
    norm = []
    for i in range(0, 42, 2):
        norm.append((features[i]     - wx) / scale)
        norm.append((features[i + 1] - wy) / scale)
    return norm


def extract_features(img_bgr):
    """Return (features[42], landmarks[21]) or (None, None) if no hand found.
    Picks the hand with the greatest wrist-to-fingertip reach (real gesture hand).
    Applies CLAHE preprocessing to handle poor lighting.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # NOTE: no flip here – raw frame is consistent with training data (american.csv)
    # Landmark x is flipped in the JSON response so the canvas matches the mirrored video.

    # CLAHE: improve contrast in dark / overexposed conditions
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = _detector.detect(mp_image)
    if not result.hand_landmarks:
        return None, None

    # Score each candidate: reach × handedness × center-priority
    best_score = -1
    best_hand  = None
    for i, lm_list in enumerate(result.hand_landmarks):
        reach = _hand_reach(lm_list)
        if reach < 0.04:
            continue
        hand_conf = result.handedness[i][0].score if result.handedness else 1.0
        center_score = _hand_center_score(lm_list)
        score = reach * (0.7 + 0.3 * hand_conf) * (0.55 + 0.45 * center_score)
        if score > best_score:
            best_score = score
            best_hand  = lm_list

    if best_hand is None:
        return None, None

    raw = []
    landmarks = []
    for lm in best_hand:
        raw.append(lm.x)
        raw.append(lm.y)
        # Flip x for canvas display so dots align with the CSS-mirrored video
        landmarks.append({"x": 1.0 - float(lm.x), "y": float(lm.y)})

    features = _normalize(raw[:42])
    return features, landmarks


def create_app(model_path=None):
    """App factory – called by run.py."""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

    app   = Flask(__name__)
    model = joblib.load(model_path)
    print(f"[ASL] Model loaded  →  {model_path}")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            payload = request.get_json(force=True)
            b64 = payload.get("image", "")
            if "," in b64:
                b64 = b64.split(",", 1)[1]

            img_bytes = base64.b64decode(b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"detected": False, "reason": "decode_failed"})

            features, landmarks = extract_features(img)
            if features is None:
                return jsonify({"detected": False, "reason": "no_hand"})

            probas  = model.predict_proba([features])[0]
            classes = model.classes_
            top3_idx = probas.argsort()[-3:][::-1]
            top3 = [{"letter": str(classes[i]).upper(), "prob": round(float(probas[i]), 3)}
                    for i in top3_idx]
            letter = top3[0]["letter"]
            confidence = top3[0]["prob"]

            return jsonify({
                "detected":    True,
                "prediction":  letter,
                "confidence":  confidence,
                "top3":        top3,
                "landmarks":   landmarks,
                "connections": HAND_CONNECTIONS,
            })

        except Exception as exc:
            return jsonify({"detected": False, "error": str(exc)}), 500

    return app


# ── Direct run (python app.py) ──────────────────────────────────────────────
if __name__ == "__main__":
    _app = create_app()
    _app.run(host="0.0.0.0", port=5000, debug=False)
