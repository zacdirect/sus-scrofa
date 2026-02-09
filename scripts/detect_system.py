#!/usr/bin/env python3
"""
Unified system detection script for GPU/CUDA availability.
Used across all AI/ML components (photoholmes, ai_detection, opencv, etc.)

This script can be called from Makefiles to:
1. Detect NVIDIA GPU hardware
2. Check NVIDIA driver installation
3. Detect CUDA version
4. Recommend appropriate PyTorch installation index URL
"""

import os
import sys
import subprocess
import shutil
import json


def check_nvidia_hardware():
    """Check if NVIDIA GPU hardware is present via lspci."""
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Check for NVIDIA in lspci output
            gpu_lines = []
            for line in result.stdout.split('\n'):
                line_lower = line.lower()
                if 'nvidia' in line_lower and ('vga' in line_lower or '3d' in line_lower):
                    gpu_lines.append(line)
            return len(gpu_lines) > 0, gpu_lines
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False, []


def check_nvidia_driver():
    """Check if NVIDIA driver is installed and working."""
    # Check nvidia-smi (most reliable)
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                return True, "nvidia-smi"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    # Check for NVIDIA kernel modules
    if os.path.exists("/proc/driver/nvidia/version"):
        return True, "kernel module"
    
    # Check for NVIDIA devices
    nvidia_devices = list(Path("/dev").glob("nvidia*"))
    if nvidia_devices:
        return True, "device files"
    
    return False, None


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available (hardware + driver)."""
    has_hardware, _ = check_nvidia_hardware()
    has_driver, _ = check_nvidia_driver()
    
    # Return True only if both hardware and driver are present
    return has_hardware and has_driver


def get_cuda_version():
    """Get CUDA version if available."""
    # Try nvcc first (most accurate)
    if shutil.which("nvcc"):
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse version from output like "release 12.4, V12.4.131"
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        parts = line.split('release')
                        if len(parts) > 1:
                            version = parts[1].strip().split(',')[0].strip()
                            return version
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    # Try to get from nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse CUDA version from nvidia-smi output
                for line in result.stdout.split('\n'):
                    if 'CUDA Version:' in line:
                        version = line.split('CUDA Version:')[1].strip().split()[0]
                        return version
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    return None


def get_pytorch_index_url(cuda_version):
    """Get appropriate PyTorch index URL based on CUDA version.
    
    Returns:
        str: PyTorch wheel index URL for CPU or CUDA-specific version
    """
    if not cuda_version:
        return "https://download.pytorch.org/whl/cpu"
    
    # Parse major.minor from version
    try:
        parts = cuda_version.split('.')
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return "https://download.pytorch.org/whl/cpu"
    
    # Map to PyTorch wheel index
    # PyTorch provides wheels for: cu118, cu121, cu124, cpu
    if major == 12:
        if minor >= 4:
            return "https://download.pytorch.org/whl/cu124"
        elif minor >= 1:
            return "https://download.pytorch.org/whl/cu121"
    elif major == 11:
        if minor >= 8:
            return "https://download.pytorch.org/whl/cu118"
    
    # Default to CPU if CUDA version not supported
    return "https://download.pytorch.org/whl/cpu"


def get_recommended_pytorch():
    """Get recommended PyTorch installation configuration.
    
    Returns:
        dict: {
            "has_gpu": bool,
            "cuda_version": str or None,
            "index_url": str,
            "backend": "CUDA" or "CPU",
            "has_hardware": bool,
            "has_driver": bool
        }
    """
    has_hardware, gpu_lines = check_nvidia_hardware()
    has_driver, driver_method = check_nvidia_driver()
    has_gpu = has_hardware and has_driver
    cuda_version = get_cuda_version() if has_gpu else None
    index_url = get_pytorch_index_url(cuda_version)
    
    return {
        "has_gpu": has_gpu,
        "has_hardware": has_hardware,
        "has_driver": has_driver,
        "driver_method": driver_method,
        "cuda_version": cuda_version,
        "index_url": index_url,
        "backend": "CUDA" if cuda_version else "CPU",
        "gpu_info": gpu_lines if has_hardware else []
    }


def main():
    """Main detection and output."""
    info = get_recommended_pytorch()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--json":
            # JSON output for programmatic use
            print(json.dumps(info, indent=2))
            
        elif arg == "--index-url":
            # Just print the index URL (for Makefile use)
            print(info["index_url"])
            
        elif arg == "--backend":
            # Just print the backend type (for Makefile use)
            print(info["backend"])
            
        elif arg == "--cuda-version":
            # Just print CUDA version (or empty if not available)
            print(info["cuda_version"] or "")
            
        elif arg == "--has-gpu":
            # Exit code 0 if GPU available, 1 if not (for Makefile conditionals)
            sys.exit(0 if info["has_gpu"] else 1)
            
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            sys.exit(1)
    else:
        # Human-readable output (default)
        print("🔍 System Detection Results")
        print("=" * 60)
        
        # Hardware detection
        if info["has_hardware"]:
            print(f"GPU Hardware:     ✓ NVIDIA GPU detected")
            for gpu_line in info["gpu_info"]:
                # Extract GPU model from lspci output
                parts = gpu_line.split(':')
                if len(parts) >= 3:
                    gpu_model = ':'.join(parts[2:]).strip()
                    print(f"                  {gpu_model}")
        else:
            print(f"GPU Hardware:     ✗ No NVIDIA GPU found")
        
        # Driver detection
        if info["has_driver"]:
            method = info["driver_method"] or "unknown"
            print(f"NVIDIA Driver:    ✓ Installed (detected via {method})")
        else:
            print(f"NVIDIA Driver:    ✗ Not installed")
        
        # CUDA version
        cuda_str = info["cuda_version"] or "Not available"
        print(f"CUDA Version:     {cuda_str}")
        
        # Recommended backend
        print(f"Recommended:      {info['backend']} backend")
        print(f"PyTorch Index:    {info['index_url']}")
        print("=" * 60)
        
        # Provide guidance based on detection
        if info["has_hardware"] and not info["has_driver"]:
            print("\n⚠️  NVIDIA GPU detected but driver not installed!")
            print("   To enable GPU acceleration:")
            print("   1. Install NVIDIA drivers:")
            print("      sudo apt install nvidia-driver-<version>")
            print("   2. Reboot your system")
            print("   3. Run 'make detect-system' again")
            print("\n   For now, PyTorch will be installed for CPU-only mode")
            print("   Inference will be slower (~2-3 seconds per image)")
            
        elif not info["has_hardware"]:
            print("\n⚠️  No NVIDIA GPU detected")
            print("   PyTorch will be installed for CPU-only mode")
            print("   Inference will be slower (~2-3 seconds per image)")
            
        elif info["has_gpu"] and info["cuda_version"]:
            print(f"\n✓ NVIDIA GPU with CUDA {info['cuda_version']} ready!")
            print("   PyTorch will be installed with GPU support")
            print("   Inference will be fast (<1 second per image)")
            
        elif info["has_driver"] and not info["cuda_version"]:
            print("\n⚠️  NVIDIA driver installed but CUDA version not detected")
            print("   Installing CPU version of PyTorch to be safe")
            print("   If you have CUDA installed, check your CUDA_HOME path")


if __name__ == "__main__":
    # Import Path only if needed
    if "--has-driver" not in sys.argv:
        from pathlib import Path
    main()
