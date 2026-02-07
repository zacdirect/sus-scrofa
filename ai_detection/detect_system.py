#!/usr/bin/env python3
"""
System detection script for AI detection module.
Detects GPU, CUDA version, and recommends appropriate PyTorch installation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


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
            for line in result.stdout.lower().split('\n'):
                if 'nvidia' in line and ('vga' in line or '3d' in line):
                    return True, result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False, None


def check_nvidia_driver():
    """Check if NVIDIA driver is installed and working."""
    # Check nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0, "nvidia-smi"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    # Check for NVIDIA kernel modules
    if os.path.exists("/proc/driver/nvidia/version"):
        return True, "kernel module"
    
    return False, None


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available (hardware + driver)."""
    has_hardware, _ = check_nvidia_hardware()
    has_driver, _ = check_nvidia_driver()
    
    # Return True only if both hardware and driver are present
    return has_hardware and has_driver


def get_cuda_version():
    """Get CUDA version if available."""
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
    """Get appropriate PyTorch index URL based on CUDA version."""
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
    """Get recommended PyTorch installation command."""
    has_gpu = check_nvidia_gpu()
    cuda_version = get_cuda_version() if has_gpu else None
    index_url = get_pytorch_index_url(cuda_version)
    
    return {
        "has_gpu": has_gpu,
        "cuda_version": cuda_version,
        "index_url": index_url,
        "backend": "CUDA" if cuda_version else "CPU"
    }


def main():
    """Main detection and output."""
    has_hardware, hardware_info = check_nvidia_hardware()
    has_driver, driver_method = check_nvidia_driver()
    cuda_version = get_cuda_version() if has_driver else None
    
    info = get_recommended_pytorch()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        # JSON output for programmatic use
        import json
        info.update({
            "has_hardware": has_hardware,
            "has_driver": has_driver,
            "driver_method": driver_method
        })
        print(json.dumps(info))
    elif len(sys.argv) > 1 and sys.argv[1] == "--index-url":
        # Just print the index URL
        print(info["index_url"])
    elif len(sys.argv) > 1 and sys.argv[1] == "--backend":
        # Just print the backend type
        print(info["backend"])
    else:
        # Human-readable output
        print("🔍 System Detection Results")
        print("=" * 40)
        
        # Hardware detection
        if has_hardware:
            print(f"GPU Hardware:     ✓ NVIDIA GPU detected")
            if hardware_info:
                for line in hardware_info.split('\n'):
                    if 'nvidia' in line.lower() and ('vga' in line.lower() or '3d' in line.lower()):
                        # Extract GPU model
                        parts = line.split(':')
                        if len(parts) >= 3:
                            gpu_model = parts[2].strip()
                            print(f"                  {gpu_model}")
        else:
            print(f"GPU Hardware:     ✗ No NVIDIA GPU found")
        
        # Driver detection
        if has_driver:
            print(f"NVIDIA Driver:    ✓ Installed (detected via {driver_method})")
        else:
            print(f"NVIDIA Driver:    ✗ Not installed")
        
        print(f"CUDA Version:     {cuda_version or 'Not available'}")
        print(f"Recommended:      {info['backend']} backend")
        print(f"PyTorch Index:    {info['index_url']}")
        print("=" * 40)
        
        # Provide guidance based on detection
        if has_hardware and not has_driver:
            print("\n⚠️  NVIDIA GPU detected but driver not installed!")
            print("   To enable GPU acceleration:")
            print("   1. Install NVIDIA drivers:")
            print("      sudo apt install nvidia-driver-<version>")
            print("   2. Reboot your system")
            print("   3. Run 'make detect-system' again")
            print("\n   For now, PyTorch will be installed for CPU-only mode")
            print("   Inference will be slower (~2-3 seconds per image)")
        elif not has_hardware:
            print("\n⚠️  No NVIDIA GPU detected")
            print("   PyTorch will be installed for CPU-only mode")
            print("   Inference will be slower (~2-3 seconds per image)")
        elif has_driver and cuda_version:
            print(f"\n✓ NVIDIA GPU with CUDA {cuda_version} ready!")
            print("   PyTorch will be installed with GPU support")
            print("   Inference will be fast (<1 second per image)")
        elif has_driver and not cuda_version:
            print("\n⚠️  NVIDIA driver installed but CUDA version not detected")
            print("   Installing CPU version of PyTorch to be safe")
            print("   If you have CUDA installed, check your CUDA_HOME path")


if __name__ == "__main__":
    main()
