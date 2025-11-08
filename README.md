# Ultra-Optimized Real-Time Vision Streaming System (YOLOv8)

A high-performance real-time video inference system using YOLOv8 that achieves minimum latency, maximum throughput, and optimal resource utilization. This system is designed for production-grade vision pipelines where efficiency, scalability, and reliability are paramount.

## Overview

This system implements a streaming inference pipeline that:
- Consumes live video input (RTSP stream or webcam)
- Performs continuous object detection with YOLOv8
- Emits inference results through a lightweight REST API
- Achieves low end-to-end latency and high sustainable FPS
- Maintains stable, predictable performance under variable load

## Architecture

The system is organized into two main modules:

### `server.py`
- Handles real-time inference using a pretrained YOLOv8 model
- Manages the streaming pipeline, performance metrics, and serving of results
- Provides REST API endpoints for inference requests
- Implements async processing with queue-based architecture for maximum throughput
- Tracks performance metrics (latency, FPS, throughput)

### `client.py`
- Ingests one or more streams (RTSP, webcam, or video files)
- Sends frames for inference to the server
- Retrieves results in real-time
- Saves results to JSON files in the specified format

## System Requirements

- Python 3.8+ (for local installation)
- OR Docker (for containerized deployment)
- CUDA-capable GPU (optional, for GPU acceleration)
- Sufficient RAM for video processing
- Network connectivity (for RTSP streams)

## Installation

### Option 1: Docker (Recommended for Easy Deployment)

```bash
# Build Docker image
docker build -t yolov8-inference-server .

# Run with Docker Compose
docker-compose up -d

# Or run directly
docker run -d -p 8000:8000 -v $(pwd)/results:/app/results yolov8-inference-server
```

See [DOCKER.md](DOCKER.md) for detailed Docker instructions.

### Option 2: Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Matrice
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 model (optional, will be downloaded automatically on first run):
```bash
# Models are downloaded automatically by ultralytics
# Available models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
```

## Usage

### Starting the Server

Start the inference server:

```bash
python server.py --host 0.0.0.0 --port 8000 --model yolov8n.pt --device auto
```

**Arguments:**
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--model`: Path to YOLOv8 model file (default: yolov8n.pt)
- `--device`: Device to run inference on - auto/cpu/cuda/mps (default: auto)

The server will automatically:
- Load the YOLOv8 model once at startup
- Start background inference processing
- Provide REST API endpoints for inference requests

### Running the Client

Process a video stream:

```bash
# RTSP stream
python client.py --server http://localhost:8000 --source rtsp://example.com/stream --stream-name cam_1

# Webcam
python client.py --server http://localhost:8000 --source 0 --stream-name webcam

# Video file
python client.py --server http://localhost:8000 --source video.mp4 --stream-name video_1

# With FPS limit
python client.py --server http://localhost:8000 --source 0 --max-fps 30
```

**Arguments:**
- `--server`: URL of the inference server (default: http://localhost:8000)
- `--source`: Video source - RTSP URL, webcam index (0, 1, etc.), or video file path
- `--stream-name`: Name of the stream (defaults to source)
- `--output-dir`: Directory to save JSON results (default: results)
- `--max-fps`: Maximum frames per second to process (optional)
- `--no-save`: Don't save results to JSON files

## API Endpoints

### POST `/inference`
Submit a frame for inference.

**Request:**
```json
{
  "frame": [[[r, g, b], ...], ...],
  "stream_name": "cam_1",
  "frame_id": 32,
  "timestamp": 1713459200.0
}
```

**Response:**
```json
{
  "timestamp": 1713459200.0,
  "frame_id": 32,
  "stream_name": "cam_1",
  "latency_ms": 20.1,
  "detections": [
    {
      "label": "person",
      "conf": 0.88,
      "bbox": [100.5, 200.3, 300.7, 400.9]
    }
  ]
}
```

### GET `/health`
Check server health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "queue_size": 0
}
```

### GET `/metrics`
Get performance metrics.

**Response:**
```json
{
  "total_inferences": 1000,
  "average_latency_ms": 25.5,
  "min_latency_ms": 15.2,
  "max_latency_ms": 45.8,
  "average_fps": 39.2,
  "current_queue_size": 0,
  "latency_history": [20.1, 22.3, 19.8, ...]
}
```

## Output Format

Results are saved to JSON files (one JSON object per line) with the following format:

```json
{
  "timestamp": 1713459200.0,
  "frame_id": 32,
  "stream_name": "cam_1",
  "latency_ms": 20.1,
  "detections": [
    {
      "label": "person",
      "conf": 0.88,
      "bbox": [100.5, 200.3, 300.7, 400.9]
    }
  ]
}
```

## Key Design Decisions

### 1. Model Loading
- Model is loaded **once** at server startup and reused throughout runtime
- This eliminates model loading overhead for each inference request
- Supports GPU acceleration when available

### 2. Async Processing Architecture
- Uses FastAPI's async capabilities for non-blocking request handling
- Implements a queue-based system for inference processing
- Separate thread pool for CPU-bound inference operations
- Prevents blocking between capture, inference, and serving

### 3. Performance Optimization
- Minimizes frame buffering to reduce latency
- Implements frame rate control on client side
- Uses efficient serialization (JSON) for API communication
- Optimized YOLOv8 inference settings (verbose=False, conf threshold)

### 4. Scalability
- Thread pool executor allows concurrent inference processing
- Queue-based architecture supports multiple concurrent streams
- Client can process multiple streams by running multiple instances
- Server can handle multiple clients simultaneously

### 5. Error Handling and Resilience
- Automatic retry strategy for HTTP requests
- Graceful error handling and logging
- Server health checks
- Timeout handling for inference requests
- Queue overflow protection

### 6. Metrics and Monitoring
- Real-time performance metrics tracking
- Latency history for analysis
- FPS calculation and monitoring
- Comprehensive logging for debugging

## Performance Considerations

### Latency Optimization
- Model loaded once at startup (no reload overhead)
- Async processing prevents blocking
- Minimal frame buffering
- Efficient queue management
- Direct frame-to-inference pipeline

### Throughput Optimization
- Thread pool for parallel inference processing
- Queue-based architecture for batch processing
- Non-blocking API endpoints
- Efficient resource utilization

### Resource Utilization
- GPU acceleration when available (automatic detection)
- CPU fallback for systems without GPU
- Configurable thread pool size
- Memory-efficient frame processing

### Stability
- Error-tolerant design with retry mechanisms
- Queue overflow protection
- Graceful degradation under high load
- Automatic recovery from errors

## Scaling Considerations

### Horizontal Scaling
- Run multiple server instances behind a load balancer
- Distribute streams across multiple servers
- Use message queue (Redis, RabbitMQ) for distributed processing

### Vertical Scaling
- Increase thread pool size for more concurrent inferences
- Use larger GPU memory for batch processing
- Optimize model size (n/s/m/l/x) based on hardware

### Multi-Stream Processing
- Run multiple client instances for different streams
- Use process pool for CPU-intensive operations
- Implement stream prioritization if needed

## Experimental Results

Results are saved to the `results/` directory as JSON files. Each line contains one inference result with:
- Timestamp
- Frame ID
- Stream name
- Latency (milliseconds)
- Detections (label, confidence, bounding box)

## Troubleshooting

### Server not starting
- Check if port 8000 is available
- Verify model file exists or can be downloaded
- Check CUDA availability for GPU acceleration

### Client connection issues
- Verify server is running and accessible
- Check network connectivity for RTSP streams
- Verify server URL is correct

### Low FPS
- Reduce input resolution
- Use smaller YOLOv8 model (yolov8n.pt)
- Enable GPU acceleration
- Increase thread pool size

### High latency
- Check network latency for RTSP streams
- Verify GPU is being used (if available)
- Reduce queue size to process faster
- Optimize frame processing

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure code follows the existing style and includes appropriate tests and documentation.

