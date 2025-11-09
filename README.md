# Ultra-Optimized Real-Time Vision Streaming System (YOLOv8)

A high-performance real-time video inference system using YOLOv8 that achieves minimum latency, maximum throughput, and optimal resource utilization. This system is designed for production-grade vision pipelines where efficiency, scalability, and reliability are paramount.

## Overview

This system implements a streaming inference pipeline that:
- Consumes live video input (RTSP stream, webcam, or video files)
- Performs continuous object detection with YOLOv8
- Emits inference results through a lightweight REST API
- Achieves low end-to-end latency and high sustainable FPS
- Maintains stable, predictable performance under variable load

## Sample Video and Results

### Test Video
A sample video has been processed with this system to demonstrate its capabilities. The test video shows person and car detection in a real-world scenario.

**Sample Video:** [Download from Google Drive](https://drive.google.com/file/d/1WlIY0izOAht43-q3uo3w1gCfRU1FS4DS/view?usp=drive_link)

### Sample Output

**Sample Results File:** `results/video_1_results.json`

You can view the complete inference output in the included JSON file, which shows:
```json
{
  "timestamp": 1762610346.809931,
  "frame_id": 0,
  "stream_name": "video_1",
  "latency_ms": 259.8,
  "detections": [
    { "label": "person", "conf": 0.69, "bbox": [2160.64, 1202.58, 2696.08, 1809.06] },
    { "label": "car", "conf": 0.66, "bbox": [6.28, 933.12, 3840.0, 2160.0] },
    { "label": "car", "conf": 0.46, "bbox": [332.84, 914.36, 738.43, 1110.83] }
  ]
}
```

## System Requirements

- Python 3.8+
- 4GB+ RAM recommended
- CUDA-capable GPU (optional, for GPU acceleration)
- Network connectivity (for RTSP streams)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Matrice
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model (optional):**
   ```bash
   # Models are downloaded automatically by ultralytics on first run
   # Available models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
   ```

## Quick Start

### 1. Start the Server

```bash
python server.py --host 0.0.0.0 --port 8000 --model yolov8n.pt --device auto
```

**Arguments:**
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--model`: Path to YOLOv8 model file (default: yolov8n.pt)
- `--device`: Device to run inference on - auto/cpu/cuda/mps (default: auto)

### 2. Run the Client

```bash
# Process webcam feed
python client.py --server http://localhost:8000 --source 0 --stream-name webcam

# Process video file
python client.py --server http://localhost:8000 --source video.mp4 --stream-name video_1

# Process RTSP stream
python client.py --server http://localhost:8000 --source rtsp://example.com/stream --stream-name cam_1

# With FPS limit
python client.py --server http://localhost:8000 --source 0 --max-fps 30

# Process the sample video
python client.py --server http://localhost:8000 --source path/to/sample_video.mp4 --stream-name video_1
```

**Arguments:**
- `--server`: URL of the inference server (default: http://localhost:8000)
- `--source`: Video source - RTSP URL, webcam index (0, 1, etc.), or video file path
- `--stream-name`: Name of the stream (defaults to source)
- `--output-dir`: Directory to save JSON results (default: results)
- `--max-fps`: Maximum frames per second to process (optional)
- `--no-save`: Don't save results to JSON files

### 3. Check Results

Results are automatically saved to `results/` directory as JSON files.

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

## Scaling and Performance

### Performance Characteristics

**Typical Performance:**
- **CPU:** YOLOv8n ~50-100ms/frame (10-20 FPS), YOLOv8s ~100-200ms/frame (5-10 FPS)
- **GPU:** YOLOv8n ~10-20ms/frame (50-100 FPS), YOLOv8s ~20-40ms/frame (25-50 FPS)

*Performance varies based on hardware, input resolution, and number of detections*

### Scaling Strategies

**Horizontal Scaling:**
- Deploy multiple server instances on different ports
- Use load balancer (Nginx) to distribute requests
- Distribute clients across multiple servers

**Vertical Scaling:**
- Increase thread pool size in `server.py` (default: 4 workers)
- Use GPU for 10x performance improvement
- Select appropriate model: yolov8n (fastest) to yolov8x (most accurate)
- Lower input resolution for faster inference

**Multi-Stream Processing:**
- Run multiple client instances connecting to same server
- Server queue handles concurrent requests from all streams
- Each stream processed independently by thread pool

### Optimization Tips

- **Latency:** Use GPU, minimize buffering, smaller model, lower resolution
- **Throughput:** Increase thread pool, deploy multiple servers, faster hardware
- **Resource Management:** Monitor memory, set queue limits, use CPU-only if GPU limited

### Monitoring

Access metrics:
```bash
curl http://localhost:8000/metrics
```

Key metrics: Latency, throughput, queue size, error rate, resource usage
