# OrbyGlasses Docker Image
# Optimized for macOS with Apple Silicon

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    portaudio19-dev \
    ffmpeg \
    espeak \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/logs data/maps models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose ports if needed (for web interface in future)
# EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import torch; print('OK')" || exit 1

# Default command
CMD ["python3", "src/main.py"]

# Note: For macOS users, run with:
# docker run --rm -it \
#   --device=/dev/video0 \
#   -v $(pwd)/data:/app/data \
#   orbglasses
