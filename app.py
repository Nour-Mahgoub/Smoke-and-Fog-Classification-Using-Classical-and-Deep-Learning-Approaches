import io
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

CLASS_NAMES = ["smoke", "non_smoke", "fog_smoke", "fog_non_smoke"]
IMG_SIZE    = (128, 128)

app = FastAPI(
    title="Smoke Classifier API",
    description=(
        "Classifies an uploaded image into one of four categories: "
        "smoke, non_smoke, fog_smoke, fog_non_smoke. "
        "Model: Custom CNN trained on IIITDMJ_Smoke dataset (test acc 96.92%), "
        "exported to ONNX for lightweight CPU inference."
    ),
    version="1.0.0",
)

# Load ONNX session once at startup
session    = ort.InferenceSession("model/smoke_classifier.onnx")
input_name = session.get_inputs()[0].name
print(f"ONNX session ready. Input: {session.get_inputs()[0].shape}")


@app.get("/")
def root():
    return {
        "name"     : "Smoke Classifier API",
        "version"  : "1.0.0",
        "model"    : "Custom CNN (ONNX) — CIE-552 Term Project",
        "classes"  : CLASS_NAMES,
        "endpoints": {
            "POST /predict": "Upload an image, receive classification JSON",
            "GET  /health" : "Liveness check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "runtime": "onnxruntime", "model_loaded": session is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()

    # Preprocess — matches Part 2 training pipeline exactly:
    # RGB, resize to 128×128, float32 normalised to [0, 1]
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = np.expand_dims(x, axis=0)        # (1, 128, 128, 3)

    t0    = time.time()
    probs = session.run(None, {input_name: x})[0][0]   # shape (4,)
    ms    = round((time.time() - t0) * 1000, 1)

    idx = int(np.argmax(probs))

    return {
        "class"        : CLASS_NAMES[idx],
        "confidence"   : round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
        "inference_ms" : ms,
    }
