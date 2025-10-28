#!/bin/bash
# Install pySLAM for OrbyGlasses
# This will give you professional SLAM with 3D visualization!

echo "🚀 Installing pySLAM with 3D Visualization..."
echo ""

# Check if pySLAM directory exists
if [ ! -d "third_party/pyslam" ]; then
    echo "❌ pySLAM directory not found at third_party/pyslam"
    echo "Please make sure the submodule is initialized"
    exit 1
fi

cd third_party/pyslam

echo "📦 Installing pySLAM dependencies..."
echo "This may take 10-15 minutes..."
echo ""

# Run pySLAM's installation script
if [ -f "scripts/install_all_venv.sh" ]; then
    chmod +x scripts/install_all_venv.sh
    ./scripts/install_all_venv.sh
else
    echo "❌ Installation script not found"
    echo "Please check the pySLAM documentation"
    exit 1
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
echo "  • Real-time 3D map viewer"
echo "  • Feature tracking display"
echo "  • Professional-grade SLAM"
echo ""

