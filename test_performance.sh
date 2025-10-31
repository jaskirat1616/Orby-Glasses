#!/bin/bash
#
# Performance Test Script for Optimized SLAM
#

echo "🚀 Testing Optimized SLAM Performance"
echo "========================================"
echo ""

# Check configuration
echo "📋 Configuration:"
grep -A 10 "^slam:" config/config.yaml | grep -E "enabled|orb_features|loop_closure|use_rerun"
echo ""

# Run SLAM for 30 seconds and capture metrics
echo "🎯 Running SLAM for 30 seconds..."
echo "Look for:"
echo "  • ⚡ Performance optimization messages"
echo "  • Feature count: ~3000"
echo "  • Matched points: >100"
echo ""

timeout 30 ./run_orby.sh 2>&1 | tee /tmp/slam_test.log | grep -E "Performance|features|FPS|matched|⚡"

echo ""
echo "✅ Test complete!"
echo ""
echo "📊 Performance Summary:"
echo "Check /tmp/slam_test.log for full details"
echo ""
echo "Expected metrics:"
echo "  • FPS: 25-35"
echo "  • Features: 2500-3000"
echo "  • Matched points: >100"
echo "  • CPU: 50-60%"
