# ==============================================
# AudioGhost AI - Backend Dockerfile (CPU Mode)
# ==============================================
# This Dockerfile builds the backend API and Celery worker
# For GPU support, use Dockerfile.gpu instead

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies including FFmpeg with shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswscale-dev \
    libswresample-dev \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version (smaller image, faster build)
# SAM-Audio requires PyTorch 2.5+ for flex_attention
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install SAM Audio
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam-audio.git

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for volumes
RUN mkdir -p uploads outputs /app/hf_cache

# Set HuggingFace cache directory
ENV HF_HOME=/app/hf_cache

# Expose API port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

