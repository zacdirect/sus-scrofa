# OpenCV Service for Ghiro
# Provides image manipulation detection via REST API
# Based on: https://github.com/demisto/dockerfiles/blob/master/docker/opencv/Dockerfile

FROM python:3.12-slim

LABEL maintainer="Ghiro Developers"
LABEL description="OpenCV-based image manipulation detection service"

# Copy requirements first for better caching
COPY requirements.txt .

# Install system dependencies and Python packages in one layer
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    libxrender1 \
    libxrender-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove \
    gcc \
    python3-dev \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the service
CMD ["python", "service.py"]

# Run the service
CMD ["python", "service.py"]
