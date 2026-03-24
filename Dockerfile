FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple torch==2.5.1 \
    && python3 -m pip install -r /app/requirements.txt

COPY main.py /app/main.py

RUN mkdir -p /data /cache/huggingface

VOLUME ["/data", "/cache/huggingface"]

ENTRYPOINT ["python3", "/app/main.py"]
CMD ["--help"]
