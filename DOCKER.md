# Docker Setup Guide

This guide explains how to build and run the Ultra-Optimized Real-Time Vision Streaming System using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

### Option 2: Using Docker Directly

#### Build the Docker Image

```bash
docker build -t yolov8-inference-server .
```

#### Run the Container

```bash
# Run server
docker run -d \
  --name yolov8-server \
  -p 8000:8000 \
  -v $(pwd)/results:/app/results \
  yolov8-inference-server

# View logs
docker logs -f yolov8-server

# Stop container
docker stop yolov8-server
docker rm yolov8-server
```

## Running the Client

The client can run on the host machine or in a separate container.

### Client on Host Machine

```bash
# Make sure server is running in Docker
# Then run client locally (requires Python and dependencies)
python client.py --server http://localhost:8000 --source 0 --stream-name webcam
```

### Client in Docker Container

```bash
# Run client in a temporary container
docker run --rm -it \
  --network host \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/client.py:/app/client.py \
  yolov8-inference-server \
  python client.py --server http://localhost:8000 --source 0 --stream-name webcam
```

## Building for Different Platforms

### Build for specific platform

```bash
# For Linux AMD64
docker build --platform linux/amd64 -t yolov8-inference-server .

# For Linux ARM64 (Apple Silicon, Raspberry Pi)
docker build --platform linux/arm64 -t yolov8-inference-server .
```

## Using GPU (NVIDIA)

If you have an NVIDIA GPU and want to use it:

### Prerequisites
- NVIDIA Docker runtime installed
- nvidia-docker2 package

### Run with GPU

```bash
docker run -d \
  --name yolov8-server \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/results:/app/results \
  yolov8-inference-server \
  python server.py --host 0.0.0.0 --port 8000 --model yolov8n.pt --device cuda
```

### Docker Compose with GPU

Update `docker-compose.yml`:

```yaml
services:
  inference-server:
    # ... existing config ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: python server.py --host 0.0.0.0 --port 8000 --model yolov8n.pt --device cuda
```

## Pushing to Docker Hub

### Build and Tag

```bash
# Build the image
docker build -t yolov8-inference-server .

# Tag for Docker Hub (replace YOUR_USERNAME)
docker tag yolov8-inference-server YOUR_USERNAME/yolov8-inference-server:latest
docker tag yolov8-inference-server YOUR_USERNAME/yolov8-inference-server:v1.0

# Login to Docker Hub
docker login

# Push to Docker Hub
docker push YOUR_USERNAME/yolov8-inference-server:latest
docker push YOUR_USERNAME/yolov8-inference-server:v1.0
```

### Pull and Run from Docker Hub

```bash
# Pull the image
docker pull YOUR_USERNAME/yolov8-inference-server:latest

# Run it
docker run -d \
  --name yolov8-server \
  -p 8000:8000 \
  -v $(pwd)/results:/app/results \
  YOUR_USERNAME/yolov8-inference-server:latest
```

## Environment Variables

You can customize the server using environment variables:

```bash
docker run -d \
  --name yolov8-server \
  -p 8000:8000 \
  -e MODEL_PATH=yolov8s.pt \
  -e DEVICE=cuda \
  -v $(pwd)/results:/app/results \
  yolov8-inference-server
```

## Volume Mounts

### Results Directory
```bash
-v $(pwd)/results:/app/results
```
This maps your local `results/` directory to the container's results directory.

### Video Files
```bash
-v $(pwd)/videos:/app/videos
```
Mount your video files directory to process videos from the container.

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs yolov8-server

# Check if port is already in use
lsof -i :8000
```

### Permission issues with volumes
```bash
# Fix permissions
sudo chown -R $USER:$USER results/
```

### Model download issues
The model will be downloaded automatically on first run. If you have network issues:
```bash
# Pre-download model and mount it
docker run -v $(pwd)/models:/app/models yolov8-inference-server \
  python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Check container status
```bash
docker ps
docker inspect yolov8-server
```

## Multi-Stage Build (Optional - for smaller images)

For production, you might want a smaller image:

```dockerfile
# Multi-stage build example
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY server.py client.py ./
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
```

## Example: Complete Workflow

```bash
# 1. Build the image
docker build -t yolov8-inference-server .

# 2. Start the server
docker-compose up -d

# 3. Check server health
curl http://localhost:8000/health

# 4. Run client (on host or in container)
python client.py --server http://localhost:8000 --source 0 --stream-name webcam

# 5. View results
ls -lh results/

# 6. Stop server
docker-compose down
```

## Production Deployment

For production, consider:
- Using a reverse proxy (nginx)
- Setting resource limits
- Using health checks
- Implementing logging
- Using secrets management
- Setting up monitoring

Example with resource limits:
```yaml
services:
  inference-server:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

