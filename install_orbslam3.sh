#!/bin/bash

# Install ORB-SLAM3 Python bindings for OrbyGlasses
# Best monocular SLAM available (2024)

echo "=========================================="
echo "Installing ORB-SLAM3 Python Bindings"
echo "Most accurate real-time monocular SLAM"
echo "=========================================="
echo ""

# Activate venv if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "[!] AUTO-INSTALL VIA pip DISABLED: Official PyPI package does NOT implement System class."
echo ""
echo "To build correct python-orbslam3 bindings, follow the instructions below."
echo ""
echo "1. Install dependencies (macOS, requires Homebrew):"
echo "   brew install cmake pkg-config eigen opencv python@3.12 openblas"
echo "   pip install numpy"
echo ""
echo "2. Clone community ORB-SLAM3 Python binding with System class:"
echo "   git clone https://github.com/uoip/python-orbslam3.git"
echo "   cd python-orbslam3"
echo "   git submodule update --init --recursive"
echo ""
echo "3. Build and install python-orbslam3 (this will take several minutes):"
echo "   python3 setup.py build"
echo "   python3 setup.py install  # (use pip install . if you want to install to venv)"
echo "   cd .. # (back to OrbyGlasses root)"
echo ""
echo "=========================================="
echo "✅ Correct ORB-SLAM3 Python bindings built/installed if no errors above!"
echo "=========================================="
echo ""
echo "If you see errors, see README or https://github.com/uoip/python-orbslam3 for help."
echo "Refer to README.md in OrbyGlasses for more info."
