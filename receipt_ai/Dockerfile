FROM python:3.10-slim

# ---- system deps for paddleocr / opencv ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- python deps ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- make PaddleOCR models be stored inside the image layer ----
ENV PADDLEOCR_HOME=/root/.paddleocr
ENV PYTHONUNBUFFERED=1

# ---- reduce CPU threads to be stable on small instances ----
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Pre-download PaddleOCR models during build (avoid first-request download)
RUN python - << 'PY'
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="german", show_log=False)
print("PaddleOCR models downloaded:", ocr)
PY

# ---- copy app code last (better docker layer cache) ----
COPY . /app

# Render will provide $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
