# Bonus A — Docker Deployment

Packages the trained smoke-classifier CNN into a self-contained REST API container.  
Send any image via HTTP POST → receive a JSON classification result in return.

---

## Folder Structure

```
docker_deployment/
├── Dockerfile          ← builds the container image
├── app.py              ← FastAPI application
├── requirements.txt    ← pinned Python dependencies
├── model/
│   └── cnn_baseline_best.h5   ← PUT YOUR MODEL FILE HERE (download from Drive)
└── Docker_README.md    ← this file
```

---

## Step 0 — Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- `cnn_baseline_best.h5` downloaded from Google Drive

---

## Step 1 — Place the Model File

Download `cnn_baseline_best.h5` from your Google Drive and put it at:

```
docker_deployment/model/cnn_baseline_best.h5
```

---

## Step 2 — Build the Docker Image

Open a terminal **inside the `docker_deployment/` folder** and run:

```bash
docker build -t smoke-classifier .
```

This will:
1. Pull `python:3.10-slim` as the base
2. Install TensorFlow, FastAPI, Uvicorn, Pillow, NumPy (~2 min on first build)
3. Copy `app.py` and the model into the image

> Subsequent builds are fast because Docker caches the pip install layer.

---

## Step 3 — Run the Container

```bash
docker run -p 8000:8000 smoke-classifier
```

The API is now live at **http://localhost:8000**

You should see:
```
Model loaded. Input: (None, 128, 128, 3)  Output: (None, 4)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 4 — Test the API

### Option A — curl (any terminal)

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/your/image.jpg"
```

Example response:
```json
{
  "class": "smoke",
  "confidence": 0.9873,
  "probabilities": {
    "smoke": 0.9873,
    "non_smoke": 0.0041,
    "fog_smoke": 0.0079,
    "fog_non_smoke": 0.0007
  },
  "inference_ms": 42.3
}
```

### Option B — Browser (interactive docs)

Open **http://localhost:8000/docs** — FastAPI auto-generates a Swagger UI.  
Click **POST /predict → Try it out → Choose File → Execute**.

### Option C — Python script

```python
import requests

url   = "http://localhost:8000/predict"
files = {"file": open("my_image.jpg", "rb")}
resp  = requests.post(url, files=files)
print(resp.json())
```

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info, class list, endpoint map |
| GET | `/health` | Liveness check |
| POST | `/predict` | Upload image → get classification JSON |
| GET | `/docs` | Swagger UI (auto-generated) |

---

## How the Preprocessing Works

The API replicates Part 2's training pipeline exactly:

| Step | Code |
|------|------|
| Load image | `PIL.Image.open(...)` |
| Convert to RGB | `.convert("RGB")` — handles grayscale and RGBA inputs |
| Resize | `.resize((128, 128))` |
| Normalise | `/ 255.0` → float32 in [0, 1] |
| Add batch dim | `np.expand_dims(x, axis=0)` → shape (1, 128, 128, 3) |
| Predict | `model.predict(x)` → softmax probabilities |

---

## Stop the Container

```bash
# Find the container ID
docker ps

# Stop it
docker stop <container_id>
```

---

## Paper Section — Deployment Strategy

*Paste into the technical paper under a subsection titled "Deployment Strategy" (§8 Option A).*

---

### Deployment Strategy

To demonstrate real-world readiness beyond the Colab notebook environment, the trained CNN
baseline (Part 2) was packaged into a self-contained Docker container exposing a REST API.
The deployment stack consists of three components:

**FastAPI backend.** The model is served via FastAPI, a high-performance Python web framework.
A single POST `/predict` endpoint accepts a multipart image upload and returns a JSON response
containing the predicted class label, confidence score, full per-class probability distribution,
and inference latency in milliseconds. The model is loaded once at container startup and reused
across all requests, avoiding repeated I/O overhead.

**Preprocessing pipeline.** The API replicates the Part 2 training pipeline exactly: the uploaded
image is converted to RGB, resized to 128×128, normalised to [0, 1] as float32, and batched before
being passed to the model. This ensures that inference results are consistent with the training
distribution.

**Docker containerisation.** A `Dockerfile` based on `python:3.10-slim` installs TensorFlow 2.17,
FastAPI, Uvicorn, Pillow, and NumPy from a pinned `requirements.txt`, then copies the application
code and model weights into the image. Pinning dependency versions guarantees that the container
produces identical results regardless of the host environment or future package updates — a key
reproducibility requirement. The server is launched via Uvicorn bound to `0.0.0.0:8000`, making
the API reachable from the host machine with a single `docker run -p 8000:8000 smoke-classifier`
command.

The complete deployment can be verified through FastAPI's auto-generated Swagger UI at
`http://localhost:8000/docs`, which provides an interactive interface for uploading images and
inspecting JSON responses without any additional client code. This setup demonstrates that the
model is portable, dependency-agnostic, and ready for integration into larger production pipelines
such as video surveillance systems, mobile backends, or cloud microservices.

---
