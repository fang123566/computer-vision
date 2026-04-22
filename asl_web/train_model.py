"""
Train an SVM classifier on the preprocessed ASL american.csv dataset.
Subsamples to SAMPLES_PER_CLASS per letter to keep training fast.
Saves the trained model to model.pkl in the same directory.
"""

import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLES_PER_CLASS = 200   # per letter; raise to 400+ for higher accuracy
RANDOM_STATE      = 42
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "american.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "model.pkl")
# ───────────────────────────────────────────────────────────────────────────────

print("Loading american.csv …")
df = pd.read_csv(CSV_PATH, header=None)
df.columns = list(range(42)) + ["label"]
df = df[df[0] != 0].reset_index(drop=True)
print(f"  american.csv valid rows : {len(df)}")

# Stratified subsample from american.csv
parts = [
    g.sample(min(SAMPLES_PER_CLASS, len(g)), random_state=RANDOM_STATE)
    for _, g in df.groupby("label")
]
df = pd.concat(parts).reset_index(drop=True)
print(f"  After subsample         : {len(df)}")

# Merge user's personal dataset if available (repeated 3× for higher weight)
USER_CSV = os.path.join(os.path.dirname(__file__), "data", "user_dataset.csv")
if os.path.exists(USER_CSV):
    dfu = pd.read_csv(USER_CSV, header=None)
    dfu.columns = list(range(42)) + ["label"]
    dfu = dfu[dfu[0] != 0].reset_index(drop=True)
    dfu["label"] = dfu["label"].str.upper()
    print(f"  User dataset            : {len(dfu)} rows  ({dfu['label'].nunique()} classes)")
    df = pd.concat([df] + [dfu] * 3, ignore_index=True)
    print(f"  Combined total          : {len(df)}")
else:
    print("  (No user_dataset.csv – training on american.csv only)")

print(f"  Classes: {sorted(df['label'].unique())}")

X = df.iloc[:, :-1].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train / Test split           : {len(X_train)} / {len(X_test)}")

print("\nTraining SVM (C=100, gamma=0.1, rbf) …")
model = SVC(C=100, gamma=0.1, kernel="rbf", probability=True, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"  Test accuracy : {acc:.4f} ({acc*100:.2f}%)\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_OUT)
print(f"Model saved → {MODEL_OUT}")
