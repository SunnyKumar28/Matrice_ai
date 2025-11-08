# Ultra-Optimized Real-Time Vision Streaming System (YOLOv8)

A high-performance real-time video inference system using YOLOv8 that achieves minimum latency, maximum throughput, and optimal resource utilization. This system is designed for production-grade vision pipelines where efficiency, scalability, and reliability are paramount.

## Overview

This system implements a streaming inference pipeline that:
- Consumes live video input (RTSP stream, webcam, or video files)
- Performs continuous object detection with YOLOv8
- Emits inference results through a lightweight REST API
- Achieves low end-to-end latency and high sustainable FPS
- Maintains stable, predictable performance under variable load

## How to Run the System

### Quick Start

**1. Start the Server**

```bash
# Using Python directly
python server.py --host 0.0.0.0 --port 8000 --model yolov8n.pt --device auto

# Or using Docker
docker-compose up -d
```

**2. Run the Client**

```bash
# Process webcam feed
python client.py --server http://localhost:8000 --source 0 --stream-name webcam

# Process video file
python client.py --server http://localhost:8000 --source video.mp4 --stream-name video_1

# Process RTSP stream
python client.py --server http://localhost:8000 --source rtsp://example.com/stream --stream-name cam_1
```

**3. Check Results**

Results are automatically saved to `results/` directory as JSON files.

### Detailed Setup Instructions

#### Prerequisites

- Python 3.8+ or Docker
- 4GB+ RAM recommended
- GPU optional (CPU works, GPU significantly faster)

#### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Matrice
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server:**
   ```bash
   python server.py --host 0.0.0.0 --port 8000
   ```

4. **In another terminal, run the client:**
   ```bash
   python client.py --server http://localhost:8000 --source 0 --stream-name webcam
   ```

### Running with Docker

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Running on Cloud (Render, AWS, etc.)

See deployment-specific documentation in the repository for cloud deployment instructions.

## Architecture Overview

### System Components

The system consists of two main components that communicate via REST API:

```
┌─────────────┐         HTTP/REST          ┌─────────────┐
│   Client    │ ──────────────────────────> │   Server    │
│  (client.py)│                             │ (server.py) │
│             │ <────────────────────────── │             │
│             │      Inference Results      │             │
└─────────────┘                             └─────────────┘
      │                                            │
      │                                            │
      ▼                                            ▼
  Video Source                                YOLOv8 Model
  (Webcam/RTSP/File)                          (Loaded Once)
```

### Component Details

#### `server.py` - Inference Server

**Responsibilities:**
- Loads YOLOv8 model once at startup (singleton pattern)
- Receives frame data via REST API
- Performs object detection inference
- Returns detection results with metadata
- Tracks performance metrics (latency, FPS, throughput)
- Manages async inference queue for concurrent processing

**Key Features:**
- FastAPI-based REST API
- Async request handling with queue-based processing
- Thread pool executor for CPU-bound inference operations
- Automatic GPU detection and utilization
- Health check and metrics endpoints

#### `client.py` - Stream Processor

**Responsibilities:**
- Captures frames from video sources (webcam, RTSP, files)
- Sends frames to inference server via HTTP
- Receives and processes inference results
- Saves results to JSON files
- Manages frame rate and connection retry logic

**Key Features:**
- Multi-source support (webcam, RTSP, video files)
- Frame rate control
- Automatic retry on connection failures
- Real-time statistics tracking
- JSON result persistence

### Data Flow

```
1. Client captures frame from video source
   ↓
2. Client converts frame to RGB format
   ↓
3. Client sends frame data to server via POST /inference
   ↓
4. Server adds frame to inference queue
   ↓
5. Background worker processes frame with YOLOv8
   ↓
6. Server returns detection results to client
   ↓
7. Client saves results to JSON file
```

### Key Design Decisions

#### 1. Model Loading Strategy
- **Decision:** Load model once at startup, reuse for all requests
- **Rationale:** Model loading is expensive (2-5 seconds). Loading once eliminates per-request overhead
- **Impact:** Reduces latency from ~5000ms to ~50ms per request

#### 2. Async Queue Architecture
- **Decision:** Use async queue with thread pool executor
- **Rationale:** 
  - FastAPI handles HTTP requests asynchronously (non-blocking)
  - Inference is CPU/GPU-bound, runs in separate threads
  - Queue buffers requests during high load
- **Impact:** Enables concurrent request handling, improves throughput

#### 3. Single Model Instance
- **Decision:** Global model variable, loaded once
- **Rationale:** Model weights are large (~6MB for nano), sharing instance saves memory
- **Impact:** Memory efficient, supports multiple concurrent streams

#### 4. JSON Output Format
- **Decision:** Save results as formatted JSON array
- **Rationale:** Human-readable, easy to parse, matches assignment specification
- **Impact:** Easy debugging and result analysis

#### 5. Client-Server Separation
- **Decision:** Separate client and server processes
- **Rationale:** 
  - Allows scaling server independently
  - Multiple clients can connect to one server
  - Client can run on different machines
- **Impact:** Horizontal scalability, distributed processing capability

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

### Error Handling and Resilience

**Automatic Retry Strategy:**
- Client retries failed requests up to 3 times
- Exponential backoff between retries
- Handles network timeouts gracefully

**Queue Overflow Protection:**
- Queue size limit (100 frames) prevents memory overflow
- Server returns 503 when queue is full
- Client can implement backpressure handling

**Health Monitoring:**
- `/health` endpoint for service monitoring
- `/metrics` endpoint for performance tracking
- Automatic recovery from transient errors


## Scaling and Performance Considerations

### Performance Characteristics

**Typical Performance (CPU):**
- YOLOv8n (nano): ~50-100ms per frame, ~10-20 FPS
- YOLOv8s (small): ~100-200ms per frame, ~5-10 FPS
- YOLOv8m (medium): ~200-400ms per frame, ~2-5 FPS

**Typical Performance (GPU):**
- YOLOv8n (nano): ~10-20ms per frame, ~50-100 FPS
- YOLOv8s (small): ~20-40ms per frame, ~25-50 FPS
- YOLOv8m (medium): ~40-80ms per frame, ~12-25 FPS

*Note: Performance varies based on hardware, input resolution, and number of detections*

### Horizontal Scaling

**Load Balancing Multiple Servers:**

1. **Deploy multiple server instances:**
   ```bash
   # Server 1
   python server.py --port 8000
   
   # Server 2
   python server.py --port 8001
   
   # Server 3
   python server.py --port 8002
   ```

2. **Use a load balancer (Nginx example):**
   ```nginx
   upstream inference_servers {
       server localhost:8000;
       server localhost:8001;
       server localhost:8002;
   }
   
   server {
       listen 80;
       location / {
           proxy_pass http://inference_servers;
       }
   }
   ```

3. **Distribute clients across servers:**
   ```bash
   # Client 1 -> Server 1
   python client.py --server http://localhost:8000 --source 0
   
   # Client 2 -> Server 2
   python client.py --server http://localhost:8001 --source 1
   ```

**Message Queue for Distributed Processing:**
- Use Redis/RabbitMQ to distribute inference tasks
- Multiple workers consume from queue
- Better for high-volume, non-real-time processing

### Vertical Scaling

**Optimize Single Server Performance:**

1. **Increase Thread Pool Size:**
   ```python
   # In server.py, modify:
   executor = ThreadPoolExecutor(max_workers=8)  # Increase from 4
   ```

2. **Use Larger GPU:**
   - More GPU memory allows batch processing
   - Multiple models can run simultaneously
   - Higher throughput for concurrent requests

3. **Model Selection:**
   - **yolov8n.pt**: Fastest, lowest accuracy, best for real-time
   - **yolov8s.pt**: Balanced speed/accuracy
   - **yolov8m.pt**: Higher accuracy, slower
   - **yolov8l/x.pt**: Highest accuracy, slowest (not recommended for real-time)

4. **Input Resolution:**
   - Lower resolution = faster inference
   - 640x480: Good balance
   - 1280x720: Higher accuracy, slower
   - 1920x1080: Best accuracy, slowest

### Multi-Stream Processing

**Running Multiple Clients:**

```bash
# Terminal 1: Process webcam 0
python client.py --server http://localhost:8000 --source 0 --stream-name webcam_0

# Terminal 2: Process webcam 1
python client.py --server http://localhost:8000 --source 1 --stream-name webcam_1

# Terminal 3: Process video file
python client.py --server http://localhost:8000 --source video.mp4 --stream-name video_1
```

**Server handles multiple streams concurrently:**
- Each stream processed independently
- Queue manages requests from all streams
- Thread pool processes frames in parallel

### Performance Optimization Strategies

**1. Latency Optimization:**
- Use GPU when available (10x faster than CPU)
- Minimize frame buffering (set `CAP_PROP_BUFFERSIZE=1`)
- Use smaller model (yolov8n.pt) for real-time applications
- Reduce input resolution if acceptable
- Run server and client on same machine to reduce network latency

**2. Throughput Optimization:**
- Increase thread pool size (more concurrent inferences)
- Use batch processing if latency allows
- Deploy multiple server instances
- Use faster hardware (GPU, more CPU cores)
- Optimize network (local network vs. internet)

**3. Resource Management:**
- Monitor memory usage (large models consume RAM)
- Set queue size limits to prevent memory overflow
- Use CPU-only mode if GPU memory is limited
- Implement request rate limiting for public APIs

**4. Scalability Patterns:**

**Pattern 1: Single Server, Multiple Clients**
```
Client 1 ──┐
Client 2 ──┼──> Server (1 instance)
Client 3 ──┘
```
- Good for: Small deployments, local networks
- Limitation: Single point of failure, limited by server capacity

**Pattern 2: Load Balanced Servers**
```
Client 1 ──┐
Client 2 ──┼──> Load Balancer ──> Server 1
Client 3 ──┘                    Server 2
                                 Server 3
```
- Good for: High availability, high throughput
- Benefit: Fault tolerance, horizontal scaling

**Pattern 3: Distributed Processing**
```
Client 1 ──> Server 1 (Stream A)
Client 2 ──> Server 2 (Stream B)
Client 3 ──> Server 3 (Stream C)
```
- Good for: Geographic distribution, dedicated resources
- Benefit: Isolation, predictable performance per stream

### Monitoring and Metrics

**Key Metrics to Monitor:**
- **Latency:** Average inference time per frame
- **Throughput:** Frames processed per second
- **Queue Size:** Number of pending inference requests
- **Error Rate:** Failed requests / total requests
- **Resource Usage:** CPU, GPU, Memory utilization

**Access Metrics:**
```bash
curl http://localhost:8000/metrics
```

**Set Up Alerts:**
- Alert if latency > 200ms
- Alert if queue size > 50
- Alert if error rate > 5%
- Alert if server health check fails

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

