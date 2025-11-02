#!/bin/bash
#
# Apply SLAM relocalization parameter fixes to pySLAM
# This script modifies pySLAM's config_parameters.py for aggressive relocalization
#

set -e

PYSLAM_CONFIG="third_party/pyslam/pyslam/config_parameters.py"

if [ ! -f "$PYSLAM_CONFIG" ]; then
    echo "❌ Error: pySLAM config not found at $PYSLAM_CONFIG"
    exit 1
fi

echo "🔧 Applying SLAM relocalization fixes to $PYSLAM_CONFIG..."

# Backup original
cp "$PYSLAM_CONFIG" "$PYSLAM_CONFIG.backup_$(date +%s)"

# Apply fixes
sed -i.tmp 's/kRelocalizationDebugAndPrintToFile = True/kRelocalizationDebugAndPrintToFile = False/' "$PYSLAM_CONFIG"
sed -i.tmp 's/kRelocalizationMinKpsMatches = 5  # ULTRA LOW/kRelocalizationMinKpsMatches = 3  # EXTREMELY LOW/' "$PYSLAM_CONFIG"
sed -i.tmp 's/0.9  # VERY LENIENT - was 0.75/0.95  # EXTREMELY LENIENT - was 0.75, then 0.9/' "$PYSLAM_CONFIG"
sed -i.tmp 's/kRelocalizationFeatureMatchRatioTestLarge = 0.95  # was 0.9/kRelocalizationFeatureMatchRatioTestLarge = 0.98  # was 0.9, then 0.95/' "$PYSLAM_CONFIG"
sed -i.tmp 's/kRelocalizationPoseOpt1MinMatches = 5  # ULTRA LOW/kRelocalizationPoseOpt1MinMatches = 3  # EXTREMELY LOW/' "$PYSLAM_CONFIG"
sed -i.tmp 's/kRelocalizationDoPoseOpt2NumInliers = 15  # ULTRA LOW/kRelocalizationDoPoseOpt2NumInliers = 8  # EXTREMELY LOW/' "$PYSLAM_CONFIG"
sed -i.tmp 's/kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 20  # LARGE WINDOW/kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 30  # HUGE WINDOW/' "$PYSLAM_CONFIG"
sed -i.tmp 's/kRelocalizationMaxReprojectionDistanceMapSearchFine = 8  # LARGE WINDOW/kRelocalizationMaxReprojectionDistanceMapSearchFine = 12  # LARGE WINDOW/' "$PYSLAM_CONFIG"

rm -f "$PYSLAM_CONFIG.tmp"

echo "✅ SLAM relocalization parameters updated successfully"
echo ""
echo "Changes applied:"
echo "  - Disabled debug logging (reduces console spam)"
echo "  - Min keypoint matches: 5 -> 3"
echo "  - Min inliers for success: 15 -> 8 (CRITICAL)"
echo "  - Search windows increased 50%"
echo "  - Match ratio thresholds: 0.9->0.95, 0.95->0.98"
echo ""
echo "These changes make relocalization MUCH more likely to succeed"
echo "in real-world monocular scenarios with limited features."
