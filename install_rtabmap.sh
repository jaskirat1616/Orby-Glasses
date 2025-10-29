#!/bin/bash
# Install RTAB-Map for OrbyGlasses
# RTAB-Map: Real-Time Appearance-Based Mapping
# Documentation: https://introlab.github.io/rtabmap/

echo "🗺️ Installing RTAB-Map for OrbyGlasses..."
echo "RTAB-Map: Real-Time Appearance-Based Mapping with loop closure detection"
echo "Documentation: https://introlab.github.io/rtabmap/"
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS. For other platforms, see:"
    echo "   https://introlab.github.io/rtabmap/"
    exit 1
fi

# Activate main venv if it exists
if [ -d "venv" ]; then
    echo "Activating main virtual environment..."
    source venv/bin/activate
fi

echo "📦 Installing RTAB-Map dependencies..."

# Install Homebrew dependencies
echo "🍺 Installing Homebrew dependencies..."
brew install cmake eigen opencv pcl g2o suite-sparse

# Install RTAB-Map Python bindings
echo "🐍 Installing RTAB-Map Python bindings..."
pip install rtabmap-python

# Alternative: Install from source if pip version doesn't work
if [ $? -ne 0 ]; then
    echo "⚠️ pip install failed, trying alternative installation..."
    
    # Install additional dependencies
    brew install qt5
    brew install sqlite3
    
    # Clone and build RTAB-Map
    echo "📥 Cloning RTAB-Map from source..."
    git clone https://github.com/introlab/rtabmap.git
    cd rtabmap
    
    # Build RTAB-Map
    echo "🛠️ Building RTAB-Map..."
    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_APP=OFF \
          -DBUILD_TOOLS=OFF \
          -DBUILD_EXAMPLES=OFF \
          -DBUILD_SHARED_LIBS=ON \
          -DCMAKE_PREFIX_PATH=/opt/homebrew \
          ..
    
    make -j$(nproc)
    sudo make install
    
    # Install Python bindings
    cd ../build/src/python
    python3 setup.py build
    python3 setup.py install
    
    cd ../../..
    rm -rf rtabmap
fi

echo ""
echo "=========================================="
echo "✅ RTAB-Map Installation Complete!"
echo "=========================================="
echo ""
echo "RTAB-Map features:"
echo "  • Real-time appearance-based mapping"
echo "  • Robust loop closure detection"
echo "  • Graph-based SLAM optimization"
echo "  • RGB-D, Stereo, and LiDAR support"
echo "  • Multi-session mapping"
echo "  • Memory management for large-scale environments"
echo ""
echo "To use RTAB-Map:"
echo "1. Edit config/config.yaml"
echo "2. Set: use_rtabmap: true"
echo "3. Run: ./run.sh"
echo ""
echo "RTAB-Map is particularly good for:"
echo "  • Long-term mapping and localization"
echo "  • Robust loop closure detection"
echo "  • Multi-session SLAM"
echo "  • Large-scale environments"
echo ""
