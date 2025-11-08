# Quick Reference Commands

## Setup (One-time)
```bash
# Navigate to project directory
cd /home/sunny-gupta/Matrice

# Activate virtual environment
source venv/bin/activate
```

## Starting the Server

### Start server (background)
```bash
cd /home/sunny-gupta/Matrice
source venv/bin/activate
python server.py --host 0.0.0.0 --port 8000 --model yolov8n.pt --device auto &
```

### Start server (foreground - see logs)
```bash
cd /home/sunny-gupta/Matrice
source venv/bin/activate
python server.py --host 0.0.0.0 --port 8000 --model yolov8n.pt --device auto
```

## Running the Client

### Use webcam
```bash
cd /home/sunny-gupta/Matrice
source venv/bin/activate
python client.py --server http://localhost:8000 --source 0 --stream-name webcam --max-fps 30
```

### Use RTSP stream
```bash
cd /home/sunny-gupta/Matrice
source venv/bin/activate
python client.py --server http://localhost:8000 --source rtsp://your-stream-url --stream-name cam_1
```

### Use video file
```bash
cd /home/sunny-gupta/Matrice
source venv/bin/activate
python client.py --server http://localhost:8000 --source video.mp4 --stream-name video_1
```

### Use webcam without FPS limit
```bash
cd /home/sunny-gupta/Matrice
source venv/bin/activate
python client.py --server http://localhost:8000 --source 0 --stream-name webcam
```


## Stopping Processes

### Stop both client and server
```bash
pkill -f "client.py"
pkill -f "server.py"
```

## Viewing Results

### List result files
```bash
ls -lh /home/sunny-gupta/Matrice/results/
```