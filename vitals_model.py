"""Tab 1 logic: input ranges, model loading, prediction, and a SHAP explanation.

The trained model is expected in models/vitals_model.joblib as a dictionary:
    {"model": fitted classifier, "features": ["HR", "RESP", "SpO2", "TEMP"],
     "classes": ["Normal", "Abnormal"], "threshold": 0.5}
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "vitals_model.joblib"

# key, label, min, max, default, step   (realistic physiological bounds, FR-3.2)
FIELDS = [
    ("HR", "Heart rate (bpm)", 30, 220, 88, 1),
    ("RESP", "Respiratory rate (breaths/min)", 4, 60, 18, 1),
    ("SpO2", "Oxygen saturation (%)", 50, 100, 97, 1),
    ("TEMP", "Body temperature (°C)", 30.0, 43.0, 37.0, 0.1),
]
NICE = {"HR": "Heart rate", "RESP": "Respiratory rate", "SpO2": "Oxygen saturation", "TEMP": "Body temperature"}


@st.cache_resource(show_spinner="Loading the vital-signs model...")
def load_bundle(path: str = str(MODEL_PATH)):
    import joblib
    return joblib.load(path)


def model_available() -> bool:
    return MODEL_PATH.exists()


def validate(values: dict):
    """Return a list of error messages (empty list = inputs are valid)."""
    errors = []
    for key, label, lo, hi, _, _ in FIELDS:
        v = values.get(key)
        if v is None or not (lo <= v <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
    return errors


def predict(bundle, values: dict):
    """Return (label, probability_abnormal, contributions DataFrame or None)."""
    features = bundle["features"]
    X = pd.DataFrame([[values[f] for f in features]], columns=features)
    model = bundle["model"]
    classes = list(bundle.get("classes", ["Normal", "Abnormal"]))
    abn = classes.index("Abnormal")
    proba = float(model.predict_proba(X)[0][abn])
    label = "Abnormal" if proba >= float(bundle.get("threshold", 0.5)) else "Normal"
    return label, proba, _explain(model, X, features)


def _explain(model, X, features):
    """SHAP contribution of every vital to this prediction (FR-3.4). None if SHAP is unavailable."""
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X)
        if isinstance(sv, list):          # older SHAP: one array per class
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # (rows, features, classes)
            sv = sv[0, :, -1]
        else:
            sv = sv[0]
        df = pd.DataFrame({"Vital sign": [NICE.get(f, f) for f in features],
                           "Value": [float(X.iloc[0][f]) for f in features],
                           "Push toward Abnormal": sv.astype(float)})
        return df.reindex(df["Push toward Abnormal"].abs().sort_values(ascending=False).index)
    except Exception:
        return None
