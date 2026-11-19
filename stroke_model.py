"""Tab 2 logic: load the YOLO stroke detector, run it on one CT slice, draw boxes,
and derive the overall slice result (Hemorrhagic / Ischemic / No Stroke)."""
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"
CLASS_NAMES = {0: "Hemorrhage", 1: "Ischemia"}
COLORS = {"Hemorrhage": (220, 38, 38), "Ischemia": (22, 163, 74)}  # red / green
DEFAULT_CONF = 0.20      # chosen on the External_Test set (sensitivity about 0.90)
IMG_SIZE = 512           # same size used for training
MAX_FILE_MB = 20
ALLOWED_EXT = (".png", ".jpg", ".jpeg", ".dcm")


@st.cache_resource(show_spinner="Loading the stroke model...")
def load_model(path: str = str(MODEL_PATH)):
    """Load once and keep in memory (NFR: models load once, not per request)."""
    from ultralytics import YOLO
    return YOLO(path)


def model_available() -> bool:
    return MODEL_PATH.exists()


def read_slice(uploaded_file) -> Image.Image:
    """Validate the upload (FR-4.2) and return an RGB image.

    DICOM files are converted with a fixed brain window (level 40 / width 80 HU).
    The model was trained on the PNG version of the dataset, so DICOM is experimental.
    """
    name = uploaded_file.name.lower()
    if not name.endswith(ALLOWED_EXT):
        raise ValueError("Unsupported file type. Please upload a PNG, JPG or DICOM (.dcm) file.")
    data = uploaded_file.getvalue()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_FILE_MB} MB.")

    if name.endswith(".dcm"):
        import pydicom
        ds = pydicom.dcmread(BytesIO(data))
        hu = ds.pixel_array.astype(np.float32)
        hu = hu * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        level, width = 40.0, 80.0
        lo, hi = level - width / 2, level + width / 2
        gray = (np.clip((hu - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB")

    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:  # corrupted or not really an image
        raise ValueError("The file could not be read as an image.") from exc


def detect(model, image: Image.Image, conf: float):
    """Run the detector. Returns a list of {'class', 'conf', 'box'} dictionaries."""
    result = model.predict(image, conf=conf, imgsz=IMG_SIZE, verbose=False)[0]
    detections = []
    for b in result.boxes:
        detections.append(
            {
                "class": CLASS_NAMES[int(b.cls[0])],
                "conf": float(b.conf[0]),
                "box": [float(v) for v in b.xyxy[0].tolist()],
            }
        )
    return sorted(detections, key=lambda d: d["conf"], reverse=True)


def draw_boxes(image: Image.Image, detections) -> Image.Image:
    """Draw every box with its class and confidence (FR-4.4)."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=max(12, out.width // 40))
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = COLORS[d["class"]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, out.width // 170))
        label = f'{d["class"]} {d["conf"]:.2f}'
        tx, ty = x1, max(0, y1 - font.size - 4) if hasattr(font, "size") else max(0, y1 - 14)
        tw = draw.textlength(label, font=font)
        th = (font.size if hasattr(font, "size") else 11) + 4
        draw.rectangle([tx, ty, tx + tw + 6, ty + th], fill=color)
        draw.text((tx + 3, ty + 1), label, fill=(255, 255, 255), font=font)
    return out


def slice_result(detections):
    """Overall result for the slice (FR-4.5, FR-4.6).

    Returns (label, confidence or None, kind) where kind is one of
    'none', 'hemorrhage', 'ischemia', 'both'.
    """
    best = {}
    for d in detections:
        best[d["class"]] = max(best.get(d["class"], 0.0), d["conf"])
    if not best:
        return "No stroke finding detected", None, "none"
    if len(best) == 2:
        return "Hemorrhagic and Ischemic findings", max(best.values()), "both"
    if "Hemorrhage" in best:
        return "Hemorrhagic Stroke", best["Hemorrhage"], "hemorrhage"
    return "Ischemic Stroke", best["Ischemia"], "ischemia"
