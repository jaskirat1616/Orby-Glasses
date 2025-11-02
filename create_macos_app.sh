#!/bin/bash
# Create macOS App Bundle for OrbyGlasses

set -e

APP_NAME="OrbyGlasses"
VERSION="0.9.0"
BUNDLE_ID="com.orbglasses.app"
BUILD_DIR="build"
APP_DIR="$BUILD_DIR/$APP_NAME.app"

echo "🚀 Creating macOS App Bundle for $APP_NAME v$VERSION..."

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Create app structure
mkdir -p "$APP_DIR/Contents"/{MacOS,Resources,Frameworks}

# Copy application files
echo "📦 Copying application files..."
cp -r src "$APP_DIR/Contents/Resources/"
cp -r config "$APP_DIR/Contents/Resources/"
cp requirements.txt "$APP_DIR/Contents/Resources/"
cp README.md "$APP_DIR/Contents/Resources/"

# Create launcher script
cat > "$APP_DIR/Contents/MacOS/$APP_NAME" <<'EOF'
#!/bin/bash
# OrbyGlasses Launcher

# Get the directory containing this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES="$DIR/../Resources"

# Set up environment
export PYTHONPATH="$RESOURCES:$PYTHONPATH"

# Check if pySLAM venv exists
PYSLAM_VENV="$HOME/.python/venvs/pyslam"
if [ -d "$PYSLAM_VENV" ]; then
    source "$PYSLAM_VENV/bin/activate"
fi

# Run OrbyGlasses
cd "$RESOURCES"
python3 src/main.py "$@"
EOF

chmod +x "$APP_DIR/Contents/MacOS/$APP_NAME"

# Create Info.plist
cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSCameraUsageDescription</key>
    <string>OrbyGlasses needs camera access for navigation assistance</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>OrbyGlasses needs microphone access for voice commands</string>
    <key>NSSupportsAutomaticTermination</key>
    <true/>
    <key>NSSupportsSuddenTermination</key>
    <false/>
</dict>
</plist>
EOF

echo "✅ App bundle created: $APP_DIR"
echo ""
echo "To install:"
echo "  cp -r $APP_DIR /Applications/"
echo ""
echo "To run:"
echo "  open $APP_DIR"
echo ""
echo "To create DMG installer:"
echo "  hdiutil create -volname \"$APP_NAME\" -srcfolder \"$APP_DIR\" -ov -format UDZO \"$BUILD_DIR/$APP_NAME-$VERSION.dmg\""
