#!/usr/bin/env python3
"""
Ultra-Optimized Real-Time Vision Streaming System - Server
Handles real-time inference using YOLOv8 model with minimal latency and maximum throughput.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance (loaded once)
model: Optional[YOLO] = None
model_lock = asyncio.Lock()

# Performance metrics
metrics = {
    'total_inferences': 0,
    'total_latency_ms': 0.0,
    'min_latency_ms': float('inf'),
    'max_latency_ms': 0.0,
    'fps_history': deque(maxlen=100),
    'latency_history': deque(maxlen=100),
}

# Inference queue for async processing
inference_queue: Queue = Queue(maxsize=100)
result_cache: Dict[str, Dict] = {}


class InferenceRequest(BaseModel):
    frame: List[List[List[int]]]  # RGB image as nested list
    stream_name: str
    frame_id: int
    timestamp: Optional[float] = None


class InferenceResponse(BaseModel):
    timestamp: float
    frame_id: int
    stream_name: str
    latency_ms: float
    detections: List[Dict]


def load_model(model_path: str = 'yolov8n.pt', device: str = 'cpu'):
    """
    Load YOLOv8 model once at startup.
    
    Args:
        model_path: Path to YOLOv8 model file
        device: Device to run inference on ('cpu', 'cuda', 'mps')
    """
    global model
    try:
        logger.info(f"Loading YOLOv8 model from {model_path} on device {device}")
        model = YOLO(model_path)
        if device == 'cuda':
            model.to('cuda')
        elif device == 'mps':
            model.to('mps')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def run_inference(frame: np.ndarray, stream_name: str, frame_id: int, 
                 request_timestamp: float) -> Dict:
    """
    Run YOLOv8 inference on a single frame.
    
    Args:
        frame: Input frame as numpy array
        stream_name: Name of the stream
        frame_id: Frame identifier
        request_timestamp: Timestamp when request was received
        
    Returns:
        Dictionary with inference results
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    inference_start = time.time()
    
    try:
        # Run inference
        results = model(frame, verbose=False, conf=0.25)
        
        inference_end = time.time()
        latency_ms = (inference_end - inference_start) * 1000
        
        # Extract detections
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    # Get confidence
                    conf = float(box.conf[0].cpu().numpy())
                    # Get class label
                    cls = int(box.cls[0].cpu().numpy())
                    label = model.names[cls]
                    
                    detections.append({
                        "label": label,
                        "conf": round(conf, 2),
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                    })
        
        # Update metrics
        metrics['total_inferences'] += 1
        metrics['total_latency_ms'] += latency_ms
        metrics['min_latency_ms'] = min(metrics['min_latency_ms'], latency_ms)
        metrics['max_latency_ms'] = max(metrics['max_latency_ms'], latency_ms)
        metrics['latency_history'].append(latency_ms)
        
        # Calculate FPS
        if len(metrics['latency_history']) > 1:
            avg_latency = np.mean(metrics['latency_history'])
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0
            metrics['fps_history'].append(fps)
        
        # Prepare response
        response = {
            "timestamp": request_timestamp if request_timestamp else time.time(),
            "frame_id": frame_id,
            "stream_name": stream_name,
            "latency_ms": round(latency_ms, 2),
            "detections": detections
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise


async def process_inference_queue(executor: ThreadPoolExecutor):
    """
    Background task to process inference requests from queue.
    """
    while True:
        try:
            # Get request from queue (non-blocking)
            try:
                request_data = inference_queue.get(timeout=0.1)
            except Empty:
                await asyncio.sleep(0.01)
                continue
            
            frame_array, stream_name, frame_id, request_timestamp, request_id = request_data
            
            # Run inference in thread pool (non-blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                run_inference,
                frame_array,
                stream_name,
                frame_id,
                request_timestamp
            )
            
            # Store result
            result_cache[request_id] = result
            
            # Mark task as done
            inference_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error processing inference queue: {e}")
            await asyncio.sleep(0.1)


# FastAPI application
app = FastAPI(title="YOLOv8 Inference Server", version="1.0.0")

# Thread pool for inference
executor = ThreadPoolExecutor(max_workers=4)


@app.on_event("startup")
async def startup_event():
    """Initialize model and start background tasks on startup."""
    # Load model if not already loaded
    global model
    if model is None:
        device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
        load_model(device=device)
    
    # Start inference queue processor
    asyncio.create_task(process_inference_queue(executor))
    logger.info("Server started and ready for inference")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    executor.shutdown(wait=True)
    logger.info("Server shutdown complete")


@app.post("/inference", response_model=InferenceResponse)
async def inference_endpoint(request: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Inference endpoint that accepts frame data and returns detection results.
    Optimized for low latency and high throughput.
    """
    try:
        # Convert frame to numpy array
        frame_array = np.array(request.frame, dtype=np.uint8)
        
        # Validate frame dimensions
        if len(frame_array.shape) != 3 or frame_array.shape[2] != 3:
            raise HTTPException(status_code=400, detail="Invalid frame format. Expected RGB image.")
        
        request_timestamp = request.timestamp if request.timestamp else time.time()
        request_id = str(uuid.uuid4())
        
        # Add to inference queue (async processing)
        try:
            inference_queue.put_nowait((
                frame_array,
                request.stream_name,
                request.frame_id,
                request_timestamp,
                request_id
            ))
        except Exception as e:
            logger.warning(f"Queue full, dropping frame {request.frame_id}")
            raise HTTPException(status_code=503, detail="Server overloaded, please retry")
        
        # Wait for result (polling with timeout)
        max_wait_time = 5.0  # 5 seconds timeout
        start_wait = time.time()
        while time.time() - start_wait < max_wait_time:
            if request_id in result_cache:
                result = result_cache.pop(request_id)
                return JSONResponse(content=result)
            await asyncio.sleep(0.001)  # 1ms polling interval
        
        # Timeout - run inference synchronously as fallback
        logger.warning(f"Timeout waiting for async result, running sync inference for frame {request.frame_id}")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            run_inference,
            frame_array,
            request.stream_name,
            request.frame_id,
            request_timestamp
        )
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in inference endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "queue_size": inference_queue.qsize()
    }


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    try:
        avg_latency = (
            metrics['total_latency_ms'] / metrics['total_inferences']
            if metrics['total_inferences'] > 0 else 0
        )
        avg_fps = 0
        if len(metrics['fps_history']) > 0:
            avg_fps = float(np.mean(list(metrics['fps_history'])))
        
        min_latency = 0
        if metrics['min_latency_ms'] != float('inf'):
            min_latency = round(metrics['min_latency_ms'], 2)
        
        # Convert deques to lists for JSON serialization
        latency_list = list(metrics['latency_history'])
        latency_history = latency_list[-10:] if len(latency_list) > 10 else latency_list
        
        return {
            "total_inferences": metrics['total_inferences'],
            "average_latency_ms": round(avg_latency, 2),
            "min_latency_ms": min_latency,
            "max_latency_ms": round(metrics['max_latency_ms'], 2),
            "average_fps": round(avg_fps, 2),
            "current_queue_size": inference_queue.qsize(),
            "latency_history": latency_history,  # Last 10 latencies
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLOv8 model")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to run inference on")
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Load model before starting server
    load_model(model_path=args.model, device=device)
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

