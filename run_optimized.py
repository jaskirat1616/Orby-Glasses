#!/usr/bin/env python3
"""
OrbyGlasses - Optimized Performance Launcher
High-performance navigation system for visually impaired users.
"""

import os
import sys
import subprocess
import time
import argparse

def check_ollama():
    """Check if Ollama is running."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def start_ollama():
    """Start Ollama service."""
    print("Starting Ollama service...")
    try:
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # Wait for service to start
        return True
    except:
        return False

def pull_required_models():
    """Pull required models for optimal performance."""
    models = ['gemma3:4b', 'moondream:latest']  # Text and Vision Language Models
    print("Pulling required models...")
    
    for model in models:
        print(f"Pulling {model}...")
        try:
            subprocess.run(['ollama', 'pull', model], check=True, timeout=600)  # Longer timeout for VLM
            print(f"✓ {model} ready")
        except subprocess.TimeoutExpired:
            print(f"⚠ {model} pull timed out - continuing")
        except subprocess.CalledProcessError:
            print(f"⚠ Failed to pull {model} - continuing")

def optimize_system():
    """Apply system optimizations."""
    print("Applying system optimizations...")
    
    # Set environment variables for performance
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Reduce MPS memory usage
    os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads
    
    print("✓ System optimizations applied")

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description='OrbyGlasses - Optimized Performance Launcher')
    parser.add_argument('--no-ollama-check', action='store_true', help='Skip Ollama check')
    parser.add_argument('--no-model-pull', action='store_true', help='Skip model pulling')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OrbyGlasses - High-Performance Navigation System")
    print("=" * 60)
    
    # Apply system optimizations
    optimize_system()
    
    # Check and start Ollama if needed
    if not args.no_ollama_check:
        if not check_ollama():
            print("Ollama not running, starting service...")
            if not start_ollama():
                print("⚠ Failed to start Ollama - continuing anyway")
        else:
            print("✓ Ollama is running")
    
    # Pull required models
    if not args.no_model_pull:
        pull_required_models()
    
    # Prepare arguments for main script
    main_args = ['python', 'src/main.py']
    if args.headless:
        main_args.append('--no-display')
    if args.save_video:
        main_args.append('--save-video')
    
    print("\nStarting OrbyGlasses with optimizations...")
    print("=" * 60)
    
    # Run the main application
    try:
        subprocess.run(main_args, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except subprocess.CalledProcessError as e:
        print(f"Error running OrbyGlasses: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
