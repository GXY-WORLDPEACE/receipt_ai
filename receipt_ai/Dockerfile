FROM python:3.10-slim

# system deps for paddleocr/opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# (Optional but recommended) reduce thread-related crashes on small machines
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PADDLEOCR_HOME=/root/.paddleocr

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
