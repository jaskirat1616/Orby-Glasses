#!/bin/bash
# Install pySLAM for OrbyGlasses
# Based on official pySLAM macOS documentation: https://github.com/luigifreda/pyslam/blob/master/docs/MAC.md

echo "🚀 Installing pySLAM with 3D Visualization..."
echo "Following official macOS guide: https://github.com/luigifreda/pyslam/blob/master/docs/MAC.md"
echo ""

# Check if pySLAM directory exists
if [ ! -d "third_party/pyslam" ]; then
    echo "❌ pySLAM directory not found at third_party/pyslam"
    echo "Please make sure the submodule is initialized"
    exit 1
fi

cd third_party/pyslam

echo "📦 Installing pySLAM dependencies for macOS..."
echo "This follows the official pySLAM macOS installation guide"
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    echo "Please follow the appropriate guide for your OS"
    exit 1
fi

# Install Homebrew dependencies first (as per official guide)
echo "🍺 Installing Homebrew dependencies..."
brew install cmake pkg-config eigen opencv g2o

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install numpy scipy matplotlib opencv-python

# Install Pangolin (required for 3D visualization)
echo "🦎 Installing Pangolin for 3D visualization..."
brew install pangolin

# Install additional dependencies
echo "📚 Installing additional dependencies..."
pip install g2o-py

# Run pySLAM's installation script
echo "🔧 Running pySLAM installation..."
if [ -f "scripts/install_all_venv.sh" ]; then
    chmod +x scripts/install_all_venv.sh
    ./scripts/install_all_venv.sh
elif [ -f "scripts/install_mac.sh" ]; then
    chmod +x scripts/install_mac.sh
    ./scripts/install_mac.sh
else
    echo "⚠️  No specific macOS script found, trying general installation..."
    if [ -f "scripts/install.sh" ]; then
        chmod +x scripts/install.sh
        ./scripts/install.sh
    else
        echo "❌ No installation script found"
        echo "Please check the pySLAM documentation"
        exit 1
    fi
fi

echo ""
echo "✅ pySLAM installation complete!"
echo ""
echo "To enable pySLAM with 3D visualization:"
echo "1. Edit config/config.yaml"
echo "2. Set: use_pyslam: true"
echo "3. Set: visualize: true"
echo "4. Run: ./run.sh"
echo ""
echo "You'll see:"
echo "  • Real-time 3D map viewer (Pangolin-based)"
echo "  • Feature tracking display"
echo "  • Professional-grade SLAM"
echo ""
echo "Based on: https://github.com/luigifreda/pyslam/blob/master/docs/MAC.md"
echo ""