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
    num_hands=2,                       # detect up to 2 hands
    min_hand_detection_confidence=0.5,  # lower threshold to catch gesture hand
)
_detector = mp.tasks.vision.HandLandmarker.create_from_options(_options)


def _hand_area(lm_list):
    """Bounding-box area of a hand landmark list – bigger = more prominent."""
    xs = [lm.x for lm in lm_list]
    ys = [lm.y for lm in lm_list]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def extract_features(img_bgr):
    """Return (features[42], landmarks[21]) or (None, None) if no hand found.
    When multiple hands are detected, picks the one with the largest bounding box.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.flip(img_rgb, 1)          # match training data convention
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = _detector.detect(mp_image)
    if not result.hand_landmarks:
        return None, None

    # Pick the hand with the largest bounding-box area (= the gesture hand)
    best = max(result.hand_landmarks, key=_hand_area)

    features  = []
    landmarks = []
    for lm in best:
        features.append(lm.x)
        features.append(lm.y)
        landmarks.append({"x": float(lm.x), "y": float(lm.y)})
    return features[:42], landmarks


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
