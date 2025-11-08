# Docker Quick Start Guide

## Build the Docker Image

```bash
docker build -t yolov8-inference-server .
```

## Run the Server

### Using Docker Compose (Easiest)

```bash
# Start server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop server
docker-compose down
```

### Using Docker Directly

```bash
# Run server
docker run -d \
  --name yolov8-server \
  -p 8000:8000 \
  -v $(pwd)/results:/app/results \
  yolov8-inference-server

# View logs
docker logs -f yolov8-server

# Stop server
docker stop yolov8-server
docker rm yolov8-server
```

## Push to Docker Hub

### 1. Tag the Image

```bash
# Replace YOUR_USERNAME with your Docker Hub username
docker tag yolov8-inference-server YOUR_USERNAME/yolov8-inference-server:latest
docker tag yolov8-inference-server YOUR_USERNAME/yolov8-inference-server:v1.0
```

### 2. Login to Docker Hub

```bash
docker login
```

### 3. Push the Image

```bash
docker push YOUR_USERNAME/yolov8-inference-server:latest
docker push YOUR_USERNAME/yolov8-inference-server:v1.0
```

## Pull and Run on Any Computer

Once pushed to Docker Hub, anyone can run it:

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

## Test the Server

```bash
# Check health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics
```

## Run Client

The client can run on the host machine (if Python is installed) or in a container:

```bash
# On host machine
python client.py --server http://localhost:8000 --source 0 --stream-name webcam
```

For more details, see [DOCKER.md](DOCKER.md).

