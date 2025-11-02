"""
GPU Speed Checker

Finds your computer's graphics chip to make OrbyGlasses faster.
"""

import torch
import platform
import subprocess
from typing import Tuple, Dict


def check_gpu_availability() -> Dict[str, any]:
    """
    Comprehensive GPU availability check.

    Returns:
        Dict with GPU information and recommendations
    """
    result = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': False,
        'mps_available': False,
        'recommended_device': 'cpu',
        'device_name': 'CPU',
        'acceleration_factor': 1.0,
        'warnings': []
    }

    # Check CUDA (NVIDIA)
    result['cuda_available'] = torch.cuda.is_available()
    if result['cuda_available']:
        result['recommended_device'] = 'cuda'
        result['device_name'] = torch.cuda.get_device_name(0)
        result['acceleration_factor'] = 10.0
        print(f"✅ CUDA GPU detected: {result['device_name']}")
        return result

    # Check MPS (Apple Silicon)
    result['mps_available'] = torch.backends.mps.is_available()
    if result['mps_available']:
        result['recommended_device'] = 'mps'
        result['device_name'] = get_apple_silicon_name()
        result['acceleration_factor'] = 5.0
        print(f"✅ Apple Silicon GPU detected: {result['device_name']}")
        return result

    # No GPU acceleration
    result['warnings'].append("No GPU acceleration available")
    result['warnings'].append("Performance will be significantly slower")
    print("⚠️  No GPU acceleration detected - using CPU")
    print("   Expected performance: 5-10 FPS (vs 20-30 FPS with GPU)")

    return result


def get_apple_silicon_name() -> str:
    """Get Apple Silicon chip name (M1, M2, M3, etc.)"""
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True,
            text=True,
            timeout=1
        )
        chip_name = result.stdout.strip()

        # Extract M1/M2/M3/M4 from brand string
        if 'Apple M' in chip_name:
            return chip_name.split('Apple ')[1].split()[0]
        return "Apple Silicon"
    except:
        return "Apple Silicon (Unknown)"


def get_optimal_device(prefer_mps: bool = True) -> str:
    """
    Get optimal device for PyTorch models.

    Args:
        prefer_mps: On macOS, prefer MPS over CPU (default True)

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'

    if torch.backends.mps.is_available() and prefer_mps:
        return 'mps'

    return 'cpu'


def verify_gpu_acceleration(device: str) -> Tuple[bool, str]:
    """
    Verify that GPU acceleration is actually working.

    Args:
        device: Device to test ('cuda', 'mps', or 'cpu')

    Returns:
        (success, message) tuple
    """
    try:
        # Create small test tensor
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)

        # Perform computation
        z = torch.matmul(x, y)

        # Verify result
        if z.shape == (100, 100):
            return True, f"✅ GPU acceleration verified on {device}"
        else:
            return False, f"❌ GPU computation returned unexpected result"

    except Exception as e:
        return False, f"❌ GPU acceleration failed: {str(e)}"


def get_memory_info(device: str) -> Dict[str, float]:
    """
    Get GPU memory information.

    Args:
        device: Device to check ('cuda' or 'mps')

    Returns:
        Dict with memory stats in MB
    """
    if device == 'cuda' and torch.cuda.is_available():
        return {
            'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
            'allocated_mb': torch.cuda.memory_allocated(0) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(0) / 1024**2
        }
    elif device == 'mps':
        # MPS doesn't provide direct memory queries
        # Estimate based on system
        try:
            result = subprocess.run(
                ['sysctl', 'hw.memsize'],
                capture_output=True,
                text=True,
                timeout=1
            )
            total_ram_bytes = int(result.stdout.split(':')[1].strip())
            total_ram_mb = total_ram_bytes / 1024**2

            # Unified memory - estimate ~70% available for GPU
            return {
                'total_mb': total_ram_mb * 0.7,
                'allocated_mb': 0,  # Can't determine
                'reserved_mb': 0
            }
        except:
            return {'total_mb': 0, 'allocated_mb': 0, 'reserved_mb': 0}
    else:
        return {'total_mb': 0, 'allocated_mb': 0, 'reserved_mb': 0}


def print_gpu_report():
    """Print comprehensive GPU acceleration report"""
    print("\n" + "="*60)
    print("GPU ACCELERATION REPORT")
    print("="*60)

    gpu_info = check_gpu_availability()

    print(f"\nPlatform: {gpu_info['platform']}")
    print(f"Python: {gpu_info['python_version']}")
    print(f"PyTorch: {gpu_info['torch_version']}")

    print(f"\nGPU Acceleration:")
    print(f"  CUDA (NVIDIA): {'✅ Available' if gpu_info['cuda_available'] else '❌ Not available'}")
    print(f"  MPS (Apple): {'✅ Available' if gpu_info['mps_available'] else '❌ Not available'}")

    print(f"\nRecommended Device: {gpu_info['recommended_device']}")
    print(f"Device Name: {gpu_info['device_name']}")
    print(f"Estimated Speedup: {gpu_info['acceleration_factor']:.1f}x vs CPU")

    # Test acceleration
    success, message = verify_gpu_acceleration(gpu_info['recommended_device'])
    print(f"\nVerification: {message}")

    # Memory info
    if gpu_info['recommended_device'] in ['cuda', 'mps']:
        mem_info = get_memory_info(gpu_info['recommended_device'])
        if mem_info['total_mb'] > 0:
            print(f"\nGPU Memory:")
            print(f"  Total: {mem_info['total_mb']:.0f} MB")
            print(f"  Allocated: {mem_info['allocated_mb']:.0f} MB")

    # Warnings
    if gpu_info['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in gpu_info['warnings']:
            print(f"  - {warning}")

    print("\n" + "="*60 + "\n")


def configure_optimal_settings(device: str) -> Dict[str, any]:
    """
    Get optimal configuration based on available hardware.

    Args:
        device: Device being used

    Returns:
        Dict with optimal settings
    """
    if device == 'cuda':
        return {
            'batch_size': 8,
            'num_workers': 4,
            'pin_memory': True,
            'use_half_precision': True,
            'compile_models': True
        }
    elif device == 'mps':
        return {
            'batch_size': 4,
            'num_workers': 2,
            'pin_memory': False,  # MPS doesn't support pinned memory
            'use_half_precision': True,
            'compile_models': False  # MPS compile support is experimental
        }
    else:  # CPU
        return {
            'batch_size': 1,
            'num_workers': 2,
            'pin_memory': False,
            'use_half_precision': False,  # CPU doesn't benefit from FP16
            'compile_models': False
        }


# Auto-detect and configure on import
_gpu_info = check_gpu_availability()
DEVICE = _gpu_info['recommended_device']
OPTIMAL_SETTINGS = configure_optimal_settings(DEVICE)


if __name__ == '__main__':
    print_gpu_report()
