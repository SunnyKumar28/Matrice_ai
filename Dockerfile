# Ultra-Optimized Real-Time Vision Streaming System - Docker Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py client.py ./

# Create results directory
RUN mkdir -p /app/results

# Expose server port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000", "--model", "yolov8n.pt", "--device", "auto"]

