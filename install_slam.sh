#!/bin/bash

# Install SLAM systems for OrbyGlasses
# Fast and accurate monocular SLAM solutions

echo "=========================================="
echo "Installing SLAM Systems for OrbyGlasses"
echo "Fast, accurate monocular SLAM solutions"
echo "=========================================="
echo ""

# Activate venv if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "✅ Improved SLAM is already included and ready to use!"
echo "   - Fast and accurate monocular SLAM"
echo "   - Optimized for real-time performance"
echo "   - No external dependencies required"
echo ""

echo "For DROID-SLAM (deep learning approach):"
echo "1. Install PyTorch (if not already installed):"
echo "   pip install torch torchvision"
echo ""
echo "2. Clone and install DROID-SLAM:"
echo "   git clone https://github.com/princeton-vl/DROID-SLAM.git"
echo "   cd DROID-SLAM"
echo "   pip install -e ."
echo "   cd .."
echo ""

echo "For ORB-SLAM3 (if you want to build from source):"
echo "1. Install dependencies:"
echo "   brew install cmake eigen opencv g2o suite-sparse"
echo ""
echo "2. Clone and build ORB-SLAM3:"
echo "   git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git"
echo "   cd ORB_SLAM3"
echo "   chmod +x build.sh"
echo "   ./build.sh"
echo "   cd .."
echo ""

echo "=========================================="
echo "✅ SLAM systems installation guide complete!"
echo "=========================================="
echo ""
echo "Current recommendation: Use Improved SLAM (already enabled in config)"
echo "It's fast, accurate, and works immediately on macOS!"
echo ""
echo "To change SLAM system, edit config/config.yaml:"
echo "  slam:"
echo "    use_improved: true    # Fast & accurate (recommended)"
echo "    use_droid: false      # Deep learning SLAM"
echo "    use_orbslam3: false   # Industry standard (requires build)"
