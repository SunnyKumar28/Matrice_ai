#!/usr/bin/env python3
"""
Ultra-Optimized Real-Time Vision Streaming System - Client
Ingests RTSP streams or video files, sends frames for inference, and saves results to JSON.
"""

import argparse
import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamClient:
    """
    Client for streaming video frames to inference server and collecting results.
    """
    
    def __init__(self, server_url: str, output_dir: str = "results", 
                 max_fps: Optional[float] = None, save_results: bool = True):
        """
        Initialize stream client.
        
        Args:
            server_url: URL of the inference server
            output_dir: Directory to save JSON results
            max_fps: Maximum frames per second to process (None for no limit)
            save_results: Whether to save results to JSON files
        """
        self.server_url = server_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_fps = max_fps
        self.save_results = save_results
        
        # Frame rate control
        self.last_frame_time = 0
        self.frame_interval = 1.0 / max_fps if max_fps else 0
        
        # Session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Statistics
        self.stats = {
            'frames_sent': 0,
            'frames_received': 0,
            'total_latency_ms': 0.0,
            'errors': 0,
            'fps_history': deque(maxlen=100),
        }
        
        # Results buffer
        self.results_buffer: List[Dict] = []
        self.results_file = None
        
    def _check_server_health(self) -> bool:
        """Check if server is healthy."""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Server health check failed: {e}")
            return False
    
    def _frame_to_list(self, frame: np.ndarray) -> List[List[List[int]]]:
        """
        Convert numpy array frame to nested list format.
        
        Args:
            frame: Frame as numpy array (BGR format from OpenCV)
            
        Returns:
            Frame as nested list (RGB format)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb.tolist()
    
    def _send_frame_for_inference(self, frame: np.ndarray, stream_name: str, 
                                  frame_id: int, timestamp: float) -> Optional[Dict]:
        """
        Send frame to inference server and get results.
        
        Args:
            frame: Frame as numpy array
            stream_name: Name of the stream
            frame_id: Frame identifier
            timestamp: Timestamp when frame was captured
            
        Returns:
            Inference result dictionary or None if error
        """
        try:
            # Convert frame to list format
            frame_list = self._frame_to_list(frame)
            
            # Prepare request
            payload = {
                "frame": frame_list,
                "stream_name": stream_name,
                "frame_id": frame_id,
                "timestamp": timestamp
            }
            
            # Send request
            request_start = time.time()
            response = self.session.post(
                f"{self.server_url}/inference",
                json=payload,
                timeout=10
            )
            request_end = time.time()
            
            if response.status_code == 200:
                result = response.json()
                end_to_end_latency = (request_end - request_start) * 1000
                
                # Update statistics
                self.stats['frames_sent'] += 1
                self.stats['frames_received'] += 1
                self.stats['total_latency_ms'] += result.get('latency_ms', 0)
                
                # Calculate FPS
                if len(self.stats['fps_history']) > 0:
                    time_diff = time.time() - self.last_frame_time
                    if time_diff > 0:
                        fps = 1.0 / time_diff
                        self.stats['fps_history'].append(fps)
                self.last_frame_time = time.time()
                
                logger.debug(f"Frame {frame_id} processed: {len(result.get('detections', []))} detections, "
                           f"latency: {result.get('latency_ms', 0):.2f}ms")
                
                return result
            else:
                logger.error(f"Inference failed for frame {frame_id}: {response.status_code} - {response.text}")
                self.stats['errors'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error sending frame {frame_id} for inference: {e}")
            self.stats['errors'] += 1
            return None
    
    def _save_result(self, result: Dict):
        """Save result to JSON file."""
        if not self.save_results:
            return
        
        self.results_buffer.append(result)
        
        # Flush buffer periodically (every 10 results or if buffer is large)
        if len(self.results_buffer) >= 10:
            self._flush_results()
    
    def _flush_results(self):
        """Flush results buffer to file."""
        if not self.results_buffer:
            return
        
        # Create results file if not exists
        if self.results_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_file = self.output_dir / f"results_{timestamp}.json"
        
        # Read existing results if file exists
        all_results = []
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        # Try to parse as JSON array first
                        try:
                            all_results = json.loads(content)
                            if not isinstance(all_results, list):
                                all_results = [all_results]
                        except json.JSONDecodeError:
                            # If not valid JSON, try JSONL format (one object per line)
                            f.seek(0)
                            for line in f:
                                line = line.strip()
                                if line:
                                    all_results.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Error reading existing results: {e}")
        
        # Append new results
        all_results.extend(self.results_buffer)
        
        # Write all results as pretty-printed JSON array with compact formatting
        # Format matches the assignment specification exactly
        with open(self.results_file, 'w') as f:
            f.write('[\n')
            for i, result in enumerate(all_results):
                f.write('  {\n')
                # Format timestamp (keep as float if decimal, otherwise int)
                timestamp = result["timestamp"]
                if isinstance(timestamp, float) and timestamp == int(timestamp):
                    timestamp = int(timestamp)
                f.write(f'    "timestamp": {timestamp},\n')
                
                f.write(f'    "frame_id": {result["frame_id"]},\n')
                f.write(f'    "stream_name": {json.dumps(result["stream_name"])},\n')
                
                # Format latency_ms (round to 1 decimal place to match example)
                latency = round(result["latency_ms"], 1)
                f.write(f'    "latency_ms": {latency},\n')
                
                f.write('    "detections": [\n')
                detections = result.get("detections", [])
                for j, det in enumerate(detections):
                    # Format bbox values (round to 2 decimal places)
                    bbox_values = [round(float(x), 2) for x in det["bbox"]]
                    bbox_str = ', '.join(str(x) for x in bbox_values)
                    
                    # Format confidence (round to 2 decimal places)
                    conf = round(float(det["conf"]), 2)
                    
                    f.write('      { ')
                    f.write(f'"label": {json.dumps(det["label"])}, ')
                    f.write(f'"conf": {conf}, ')
                    f.write(f'"bbox": [{bbox_str}]')
                    f.write(' }')
                    if j < len(detections) - 1:
                        f.write(',')
                    f.write('\n')
                f.write('    ]\n')
                f.write('  }')
                if i < len(all_results) - 1:
                    f.write(',')
                f.write('\n')
            f.write(']\n')
        
        logger.debug(f"Saved {len(self.results_buffer)} results to {self.results_file} (total: {len(all_results)})")
        self.results_buffer.clear()
    
    def process_stream(self, source: str, stream_name: Optional[str] = None):
        """
        Process video stream from RTSP URL, webcam, or video file.
        
        Args:
            source: RTSP URL, webcam index (0, 1, etc.), or video file path
            stream_name: Name of the stream (defaults to source)
        """
        if stream_name is None:
            stream_name = Path(source).stem if Path(source).exists() else source
        
        # Check server health
        if not self._check_server_health():
            logger.error("Server is not healthy. Please start the server first.")
            return
        
        logger.info(f"Starting stream processing: {source} as '{stream_name}'")
        
        # Open video source
        if source.isdigit():
            # Webcam
            cap = cv2.VideoCapture(int(source))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
        elif source.startswith(('rtsp://', 'http://', 'https://')):
            # RTSP stream
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # Video file
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return
        
        frame_id = 0
        start_time = time.time()
        
        try:
            while True:
                # Frame rate control
                if self.max_fps:
                    elapsed = time.time() - self.last_frame_time
                    if elapsed < self.frame_interval:
                        time.sleep(self.frame_interval - elapsed)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_id} from {source}")
                    # For video files, break on EOF
                    if not source.startswith(('rtsp://', 'http://', 'https://')) and not source.isdigit():
                        break
                    continue
                
                # Get timestamp
                timestamp = time.time()
                
                # Send frame for inference
                result = self._send_frame_for_inference(frame, stream_name, frame_id, timestamp)
                
                if result:
                    # Save result
                    self._save_result(result)
                
                frame_id += 1
                
                # Log progress periodically
                if frame_id % 100 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_id / elapsed_time if elapsed_time > 0 else 0
                    avg_fps = np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0
                    logger.info(f"Processed {frame_id} frames, "
                              f"FPS: {fps:.2f}, Avg FPS: {avg_fps:.2f}, "
                              f"Errors: {self.stats['errors']}")
        
        except KeyboardInterrupt:
            logger.info("Stream processing interrupted by user")
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
        finally:
            # Flush remaining results
            self._flush_results()
            
            # Release capture
            cap.release()
            
            # Print final statistics
            self._print_statistics()
    
    def _print_statistics(self):
        """Print processing statistics."""
        logger.info("=" * 50)
        logger.info("Processing Statistics:")
        logger.info(f"  Frames sent: {self.stats['frames_sent']}")
        logger.info(f"  Frames received: {self.stats['frames_received']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        if self.stats['frames_received'] > 0:
            avg_latency = self.stats['total_latency_ms'] / self.stats['frames_received']
            logger.info(f"  Average latency: {avg_latency:.2f}ms")
        if self.stats['fps_history']:
            avg_fps = np.mean(self.stats['fps_history'])
            logger.info(f"  Average FPS: {avg_fps:.2f}")
        logger.info("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Client")
    parser.add_argument("--server", default="http://localhost:8000", 
                       help="URL of the inference server")
    parser.add_argument("--source", required=True,
                       help="Video source: RTSP URL, webcam index (0, 1, etc.), or video file path")
    parser.add_argument("--stream-name", default=None,
                       help="Name of the stream (defaults to source)")
    parser.add_argument("--output-dir", default="results",
                       help="Directory to save JSON results")
    parser.add_argument("--max-fps", type=float, default=None,
                       help="Maximum frames per second to process")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to JSON files")
    
    args = parser.parse_args()
    
    # Create client
    client = StreamClient(
        server_url=args.server,
        output_dir=args.output_dir,
        max_fps=args.max_fps,
        save_results=not args.no_save
    )
    
    # Process stream
    client.process_stream(args.source, args.stream_name)


if __name__ == "__main__":
    main()

