FROM python:3.10-slim

WORKDIR /app

# Install dependencies — onnxruntime is ~7 MB, total pip download < 30 MB
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

COPY app.py .
COPY model/ model/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
