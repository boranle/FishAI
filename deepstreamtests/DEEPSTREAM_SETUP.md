# NVIDIA DeepStream Setup Guide for Jetson Orin Nano

## Prerequisites

- **Hardware**: NVIDIA Jetson Orin Nano
- **OS**: JetPack 6.2
- **Storage**: At least 10GB free space
- **Power**: Adequate power supply recommended (25W+)

## Installation Steps

### 1. Update System
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 2. Install JetPack 6.2 (if not already installed)
```bash
# Using NVIDIA SDKManager
# Download from: https://developer.nvidia.com/nvidia-sdk-manager

# OR use apt-get for quick installation
sudo apt-get install nvidia-jetpack
```

### 3. Install DeepStream SDK
```bash
# Add NVIDIA DeepStream repository
sudo apt-get install nvidia-deepstream
# or
sudo apt-get install deepstream-6.3
```

### 4. Verify Installation
```bash
nvidia-smi  # Check GPU
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

### 5. Install Python Dependencies
```bash
pip3 install pyyaml gi
# or
pip install -r requirements.txt
```

### 6. Run the DeepStream Program

#### Option A: RTSP Stream Input
```bash
# Modify RTSP_URI in deepstream_inference.py to your camera URL
python3 deepstream_inference.py
```

#### Option B: Video File Input
```bash
# Edit deepstream_inference.py and change:
# RTSP_URI = "file:///path/to/video.mp4"
python3 deepstream_inference.py
```

#### Option C: Using Config File
```bash
deepstream-app -c deepstream_config.txt
```

## Configuration

### Key Parameters in `deepstream_inference.py`

```python
RTSP_URI = "rtsp://IP:PORT/stream"      # Your camera RTSP stream
DETECT_MODEL = "resnet10"               # Detection model type
CONF_THRESHOLD = 0.5                    # Confidence threshold
OUTPUT_VIDEO = False                    # Save output to file
VIDEO_OUTPUT_PATH = "/tmp/output.mkv"   # Output file path
```

### Available Detection Models

1. **ResNet-10** (default, fast)
   - Good for real-time processing
   - Lower accuracy

2. **YOLOv3/v4/v5**
   - Better accuracy
   - Slower inference
   - Requires custom model engine files

3. **Faster R-CNN**
   - High accuracy
   - Slower inference

## Performance Optimization

### For Jetson Orin Nano:

1. **Enable Max Performance Mode**
```bash
sudo nvpmodel -m 0  # Maximum performance
sudo jetson_clocks.sh  # Boost clocks
```

2. **Use INT8 Quantization**
   - Modify model engine to use int8 precision
   - ~2-3x faster inference with minimal accuracy loss

3. **Batch Processing**
   - Increase batch-size in pipeline for multiple streams
   - Monitor memory usage

### Monitoring Performance
```bash
jtop  # Install: pip3 install jetson-stats

# Or use tegrastats
tegrastats
```

## Troubleshooting

### Error: "Failed to create GStreamer element"
```bash
# Ensure GStreamer plugins are installed
sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

### Error: RTSP connection failed
```bash
# Test connection directly
gst-launch-1.0 rtspsrc location="rtsp://IP:PORT/stream" ! fakesink
```

### Out of Memory
```bash
# Reduce batch size, model size, or input resolution
# Edit streammux properties in deepstream_inference.py
streammux.set_property("width", 1280)      # Reduce from 1920
streammux.set_property("height", 720)      # Reduce from 1080
```

### GPU Memory Issues
```bash
# Check available VRAM
$ nvidia-smi
# Typical Jetson Orin Nano: 8GB shared CUDA

# Monitor during runtime
watch -n 1 nvidia-smi
```

## Performance Recommendations

| Setting | Orin Nano | Notes |
|---------|-----------|-------|
| Max Power | 25W | Adequate cooling required |
| Batch Size | 1-2 | Limited by 8GB VRAM |
| Input Resolution | 1280x720 | Balance between accuracy and speed |
| FPS Target | 30 | Depends on model and resolution |
| Model Precision | INT8 | For speed on Orin Nano |

## Advanced: Using Custom Models

### Convert TensorFlow/PyTorch to TensorRT
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --int8
```

### Update Config File
```bash
# In deepstream_config.txt
[primary-gie]
model-engine-file=/path/to/custom_model.engine
```

## Resources

- [NVIDIA DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [Jetson Orin Nano Specs](https://developer.nvidia.com/embedded/jetson-orin-nano)
- [JetPack 6.2 Release Notes](https://docs.nvidia.com/jetson/jetpack/)
- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)

## Support

For issues, check:
1. Stack Overflow: `nvidia-deepstream jetson-orin-nano`
2. NVIDIA Developer Forum: forums.developer.nvidia.com
3. DeepStream GitHub Issues: github.com/NVIDIA-AI-IOT/deepstream_reference_apps
