#!/usr/bin/env python3
"""
OrbyGlasses Performance Benchmark
Tests and validates performance improvements.
"""

import time
import numpy as np
import cv2
import sys
import os
from typing import Dict, List
import threading
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import ConfigManager, PerformanceMonitor
from detection import DetectionPipeline

class PerformanceBenchmark:
    """Benchmark OrbyGlasses performance."""
    
    def __init__(self):
        self.config = ConfigManager("config/config.yaml")
        self.perf_monitor = PerformanceMonitor()
        self.detection_pipeline = DetectionPipeline(self.config)
        
        # Test frame
        self.test_frame = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        
    def benchmark_detection(self, num_runs: int = 50) -> Dict:
        """Benchmark object detection performance."""
        print("Benchmarking object detection...")
        
        times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            detections = self.detection_pipeline.detector.detect(self.test_frame)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if i % 10 == 0:
                print(f"  Run {i+1}/{num_runs}: {times[-1]:.2f}ms")
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000.0 / np.mean(times)
        }
    
    def benchmark_depth_estimation(self, num_runs: int = 20) -> Dict:
        """Benchmark depth estimation performance."""
        print("Benchmarking depth estimation...")
        
        times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            depth_map = self.detection_pipeline.depth_estimator.estimate_depth(self.test_frame)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if i % 5 == 0:
                print(f"  Run {i+1}/{num_runs}: {times[-1]:.2f}ms")
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000.0 / np.mean(times)
        }
    
    def benchmark_full_pipeline(self, num_runs: int = 30) -> Dict:
        """Benchmark full processing pipeline."""
        print("Benchmarking full pipeline...")
        
        times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            
            # Detection
            detections = self.detection_pipeline.detector.detect(self.test_frame)
            
            # Depth (every 3rd frame)
            if i % 3 == 0:
                depth_map = self.detection_pipeline.depth_estimator.estimate_depth(self.test_frame)
            else:
                depth_map = None
            
            # Add depth to detections
            if depth_map is not None:
                for detection in detections:
                    bbox = detection['bbox']
                    depth = self.detection_pipeline.depth_estimator.get_depth_at_bbox(depth_map, bbox)
                    detection['depth'] = depth
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if i % 10 == 0:
                print(f"  Run {i+1}/{num_runs}: {times[-1]:.2f}ms")
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000.0 / np.mean(times)
        }
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage."""
        import psutil
        import gc
        
        print("Benchmarking memory usage...")
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run detection multiple times
        for _ in range(10):
            detections = self.detection_pipeline.detector.detect(self.test_frame)
            depth_map = self.detection_pipeline.depth_estimator.estimate_depth(self.test_frame)
            gc.collect()  # Force garbage collection
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'baseline_mb': baseline_memory,
            'peak_mb': peak_memory,
            'usage_mb': peak_memory - baseline_memory
        }
    
    def run_full_benchmark(self):
        """Run complete performance benchmark."""
        print("=" * 60)
        print("OrbyGlasses Performance Benchmark")
        print("=" * 60)
        
        results = {}
        
        # Detection benchmark
        results['detection'] = self.benchmark_detection()
        
        # Depth estimation benchmark
        results['depth'] = self.benchmark_depth_estimation()
        
        # Full pipeline benchmark
        results['pipeline'] = self.benchmark_full_pipeline()
        
        # Memory usage benchmark
        results['memory'] = self.benchmark_memory_usage()
        
        # Print results
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        print(f"\nObject Detection:")
        print(f"  Mean: {results['detection']['mean_ms']:.2f}ms")
        print(f"  FPS: {results['detection']['fps']:.1f}")
        print(f"  Std Dev: {results['detection']['std_ms']:.2f}ms")
        
        print(f"\nDepth Estimation:")
        print(f"  Mean: {results['depth']['mean_ms']:.2f}ms")
        print(f"  FPS: {results['depth']['fps']:.1f}")
        print(f"  Std Dev: {results['depth']['std_ms']:.2f}ms")
        
        print(f"\nFull Pipeline:")
        print(f"  Mean: {results['pipeline']['mean_ms']:.2f}ms")
        print(f"  FPS: {results['pipeline']['fps']:.1f}")
        print(f"  Std Dev: {results['pipeline']['std_ms']:.2f}ms")
        
        print(f"\nMemory Usage:")
        print(f"  Baseline: {results['memory']['baseline_mb']:.1f}MB")
        print(f"  Peak: {results['memory']['peak_mb']:.1f}MB")
        print(f"  Usage: {results['memory']['usage_mb']:.1f}MB")
        
        # Performance assessment
        print(f"\n" + "=" * 60)
        print("PERFORMANCE ASSESSMENT")
        print("=" * 60)
        
        detection_fps = results['detection']['fps']
        pipeline_fps = results['pipeline']['fps']
        
        if pipeline_fps >= 20:
            print("✓ EXCELLENT: Real-time performance achieved")
        elif pipeline_fps >= 15:
            print("✓ GOOD: Near real-time performance")
        elif pipeline_fps >= 10:
            print("⚠ ACCEPTABLE: Moderate performance")
        else:
            print("✗ POOR: Performance needs improvement")
        
        print(f"\nTarget FPS: 30 | Achieved: {pipeline_fps:.1f}")
        print(f"Detection FPS: {detection_fps:.1f}")
        
        return results

def main():
    """Run performance benchmark."""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_full_benchmark()
    
    # Save results
    import json
    with open('performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to performance_results.json")

if __name__ == "__main__":
    main()
