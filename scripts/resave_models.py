#!/usr/bin/env python3
"""
Script: resave_models.py
- Loads model and scaler from default locations (joblib first, pickle fallback).
- Runs a tiny smoke test (dummy input) to ensure objects are functional.
- Re-saves objects using joblib.dump() to new files in `models/`.

Usage:
    python3 scripts/resave_models.py

This will create `models/isolation_forest.joblib` and
`models/scaler_if.joblib` if successful.
"""
import os
import sys
import json
import numpy as np
import joblib
import pickle

MODEL_PATH = os.path.join('models', 'isolation_forest.pkl')
SCALER_PATH = os.path.join('models', 'scaler_if.pkl')
OUT_MODEL = os.path.join('models', 'isolation_forest.joblib')
OUT_SCALER = os.path.join('models', 'scaler_if.joblib')

def load_obj(path):
    obj = None
    if not os.path.exists(path):
        print(f"[WARN] path not found: {path}")
        return None
    # try joblib first
    try:
        obj = joblib.load(path)
        print(f"Loaded via joblib: {path}")
        return obj
    except Exception as e:
        print(f"joblib.load failed for {path}: {e}")
    # fallback to pickle
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Loaded via pickle: {path}")
        return obj
    except Exception as e:
        print(f"pickle.load failed for {path}: {e}")
        return None


def smoke_test_model_scaler(model, scaler):
    # Create a plausible dummy feature vector size based on scaler or model
    if scaler is not None:
        try:
            n_features = scaler.scale_.shape[0]
        except Exception:
            # fallback guess
            n_features = 11  # minimal guess (one axis)
    else:
        # try model attribute
        try:
            n_features = model.n_features_in_
        except Exception:
            n_features = 11
    X = np.zeros((1, n_features), dtype=float)
    ok = True
    if scaler is not None:
        try:
            Xs = scaler.transform(X)
            print(f"Scaler.transform OK: produced shape {Xs.shape}")
        except Exception as e:
            print(f"Scaler.transform failed: {e}")
            ok = False
    if model is not None:
        try:
            # try predict or score_samples depending on availability
            if hasattr(model, 'predict'):
                p = model.predict(X)
                print(f"Model.predict OK: {p}")
            elif hasattr(model, 'score_samples'):
                s = model.score_samples(X)
                print(f"Model.score_samples OK: {s}")
            else:
                print("Model has no predict/score_samples method; skipping smoke test")
        except Exception as e:
            print(f"Model inference failed: {e}")
            ok = False
    return ok


def main():
    print("Resave models script starting...")
    model = load_obj(MODEL_PATH)
    scaler = load_obj(SCALER_PATH)

    ok = smoke_test_model_scaler(model, scaler)
    if not ok:
        print("[WARN] Smoke test failed; you may need to install matching scikit-learn or inspect model files.")

    # Attempt to re-save using joblib to new files (do not overwrite originals)
    try:
        if model is not None:
            joblib.dump(model, OUT_MODEL)
            print(f"Re-saved model -> {OUT_MODEL}")
        else:
            print("No model to re-save")
    except Exception as e:
        print(f"Failed to joblib.dump model: {e}")

    try:
        if scaler is not None:
            joblib.dump(scaler, OUT_SCALER)
            print(f"Re-saved scaler -> {OUT_SCALER}")
        else:
            print("No scaler to re-save")
    except Exception as e:
        print(f"Failed to joblib.dump scaler: {e}")

    print("Done.")

if __name__ == '__main__':
    main()
