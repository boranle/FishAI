#!/bin/bash
# DeepStream Installation Script for Jetson Orin Nano
# JetPack 6.2
# 
# Usage: sudo bash install_deepstream.sh

set -e

echo "======================================"
echo "DeepStream Installation for Jetson"
echo "JetPack 6.2"
echo "======================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ This script must be run with sudo"
    exit 1
fi

echo ""
echo "📦 Step 1: Updating package manager..."
apt-get update
apt-get upgrade -y

echo ""
echo "📦 Step 2: Installing GStreamer dependencies..."
apt-get install -y \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools

echo ""
echo "📦 Step 3: Installing NVIDIA DeepStream SDK..."
apt-get install -y nvidia-deepstream

echo ""
echo "📦 Step 4: Installing Python GObject bindings..."
apt-get install -y \
    python3-gi \
    gir1.2-gstreamer-1.0

echo ""
echo "📦 Step 5: Installing Python development tools..."
apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential

echo ""
echo "📦 Step 6: Installing Jetson tools..."
apt-get install -y jetson-utils jetson-stats

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Reboot the system: sudo reboot"
echo "2. Enable max performance mode:"
echo "   sudo nvpmodel -m 0"
echo "   sudo jetson_clocks.sh"
echo "3. Run the DeepStream examples:"
echo "   python3 deepstream_inference.py"
echo "4. Or try the advanced YOLO version:"
echo "   python3 deepstream_yolo_advanced.py"
echo ""
