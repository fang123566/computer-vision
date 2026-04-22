"""
Entry point.  Usage:
    python run.py          # auto-trains if model.pkl missing, then starts server
    python run.py --train  # force re-train even if model.pkl exists
Open http://localhost:5000 in browser after starting.
"""

import os, sys, subprocess, argparse

BASE  = os.path.dirname(os.path.abspath(__file__))
VENV  = os.path.join(BASE, "..", ".venv", "Scripts", "python.exe")
PYTHON = os.path.normpath(VENV) if os.path.exists(os.path.normpath(VENV)) else sys.executable
MODEL = os.path.join(BASE, "model.pkl")

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="Force re-train model")
parser.add_argument("--port",  type=int, default=5000)
args = parser.parse_args()

if args.train or not os.path.exists(MODEL):
    print("=" * 55)
    print("  Training model (first run) …")
    print("=" * 55)
    ret = subprocess.run(
        [PYTHON, os.path.join(BASE, "train_model.py")],
        check=False
    )
    if ret.returncode != 0:
        print("\n[ERROR] Training failed. Check error above.")
        sys.exit(1)

print(f"\n{'='*55}")
print(f"  ASL Recognition Server")
print(f"  http://localhost:{args.port}")
print(f"{'='*55}\n")

os.chdir(BASE)
from app import create_app
app = create_app(MODEL)
app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
