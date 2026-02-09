# GPU/CUDA Detection System

## Overview

Sus Scrofa uses a unified system detection approach to automatically detect GPU/CUDA availability and configure appropriate PyTorch installations across all AI/ML components.

## Unified Detection Script

**Location:** `scripts/detect_system.py`

This Python script provides comprehensive GPU/CUDA detection:

- **Hardware Detection**: Checks for NVIDIA GPU via `lspci`
- **Driver Detection**: Verifies NVIDIA driver installation via `nvidia-smi`, kernel modules, or device files
- **CUDA Version**: Detects CUDA version from `nvcc` or `nvidia-smi`
- **PyTorch Index**: Returns appropriate PyTorch wheel index URL based on detected CUDA version

### Usage

```bash
# Human-readable output with recommendations
python3 scripts/detect_system.py

# Get PyTorch index URL (for Makefile use)
python3 scripts/detect_system.py --index-url

# Get backend type: "CPU" or "CUDA"
python3 scripts/detect_system.py --backend

# Get CUDA version (empty if not available)
python3 scripts/detect_system.py --cuda-version

# Check if GPU available (exit code 0=yes, 1=no)
python3 scripts/detect_system.py --has-gpu

# JSON output (for programmatic use)
python3 scripts/detect_system.py --json
```

### Example Output

```bash
$ python3 scripts/detect_system.py
🔍 System Detection Results
============================================================
GPU Hardware:     ✓ NVIDIA GPU detected
                  NVIDIA Corporation TU117GLM [Quadro T1000 Mobile] (rev a1)
NVIDIA Driver:    ✗ Not installed
CUDA Version:     Not available
Recommended:      CPU backend
PyTorch Index:    https://download.pytorch.org/whl/cpu
============================================================

⚠️  NVIDIA GPU detected but driver not installed!
   To enable GPU acceleration:
   1. Install NVIDIA drivers:
      sudo apt install nvidia-driver-<version>
   2. Reboot your system
   3. Run 'make detect-system' again

   For now, PyTorch will be installed for CPU-only mode
   Inference will be slower (~2-3 seconds per image)
```

## Makefile Integration

### Main Makefile

The root `Makefile` provides a convenient detection target:

```bash
make detect-system
```

All AI/ML setup targets use the unified detection script:

#### Photoholmes Setup

```makefile
photoholmes-setup:
    @INDEX_URL=$$($(SYSTEM_PYTHON) scripts/detect_system.py --index-url); \
    BACKEND=$$($(SYSTEM_PYTHON) scripts/detect_system.py --backend); \
    echo "Backend: $$BACKEND"; \
    echo "PyTorch Index: $$INDEX_URL"; \
    $(PIP) install --index-url $$INDEX_URL torch torchvision; \
    ...
```

**Result**: Automatically installs CPU-only torch (~190MB) if no GPU, or CUDA-enabled torch if GPU detected.

#### AI Detection Setup

```makefile
ai-setup:
    $(MAKE) -C ai_detection setup
```

The `ai_detection/Makefile` also uses the unified script:

```makefile
install: venv
    @INDEX_URL=$$($(SYSTEM_PYTHON) ../scripts/detect_system.py --index-url); \
    BACKEND=$$($(SYSTEM_PYTHON) ../scripts/detect_system.py --backend); \
    $(PIP) install --extra-index-url $$INDEX_URL torch>=2.0.0 torchvision>=0.15.0; \
    ...
```

## PyTorch Index Mapping

The detection script maps CUDA versions to PyTorch wheel indexes:

| CUDA Version | PyTorch Index |
|--------------|---------------|
| 12.4+ | `https://download.pytorch.org/whl/cu124` |
| 12.1-12.3 | `https://download.pytorch.org/whl/cu121` |
| 11.8+ | `https://download.pytorch.org/whl/cu118` |
| None/Unsupported | `https://download.pytorch.org/whl/cpu` |

## Detection Logic

The script follows this decision tree:

```
1. Check for NVIDIA GPU hardware (lspci)
   ├─ No → Install CPU PyTorch
   └─ Yes → Continue
        
2. Check for NVIDIA driver (nvidia-smi, kernel modules, device files)
   ├─ No → Install CPU PyTorch (warn user to install driver)
   └─ Yes → Continue
        
3. Detect CUDA version (nvcc, nvidia-smi)
   ├─ None → Install CPU PyTorch (warn about missing CUDA)
   └─ X.Y → Map to appropriate PyTorch CUDA index
```

## Benefits

### Automatic Configuration

No manual configuration needed. The system automatically:
- Detects GPU availability
- Chooses optimal PyTorch version
- Provides clear guidance when drivers missing

### Consistent Across Components

All AI/ML components use the same detection:
- Photoholmes forgery detection
- AI Detection (SPAI, SDXL)
- Future GPU-accelerated plugins

### Resource Efficiency

- **CPU systems**: Installs lightweight ~190MB torch
- **GPU systems**: Installs CUDA-enabled ~2GB+ torch (faster inference)

### User Guidance

When GPU hardware detected but driver missing:
```
⚠️  NVIDIA GPU detected but driver not installed!
   To enable GPU acceleration:
   1. Install NVIDIA drivers:
      sudo apt install nvidia-driver-<version>
   2. Reboot your system
   3. Run 'make detect-system' again
```

## Troubleshooting

### "No NVIDIA GPU detected" but I have one

**Cause**: `lspci` not showing NVIDIA GPU

**Solutions**:
- Run `lspci | grep -i nvidia` to verify
- Update PCI IDs: `sudo update-pciids`
- Check BIOS settings (ensure GPU not disabled)

### "NVIDIA driver not installed" but nvidia-smi works

**Cause**: Script checks nvidia-smi, kernel modules, and device files

**Solutions**:
- Run `nvidia-smi` manually to verify
- Check `/proc/driver/nvidia/version` exists
- Check `/dev/nvidia*` devices exist

### CUDA version detected incorrectly

**Cause**: Multiple CUDA installations or mismatched versions

**Solutions**:
- Run `nvcc --version` to see compiler version
- Run `nvidia-smi` to see runtime version
- Use the version reported by `nvidia-smi` (more reliable)

### PyTorch still installing CUDA version on CPU system

**Cause**: Likely not using the Makefile targets

**Solution**: Always use `make photoholmes-setup` or `make ai-setup`, never install torch directly with pip

## Future Enhancements

Potential improvements to the detection system:

1. **Multi-GPU Support**: Detect number of GPUs, select optimal
2. **ROCm Support**: Add AMD GPU detection (ROCm instead of CUDA)
3. **Apple Silicon**: Detect M1/M2 for Metal acceleration
4. **Memory Detection**: Check GPU VRAM, warn if insufficient
5. **Conda Integration**: Support conda environments in addition to venv
6. **Cache Detection Results**: Avoid repeated `lspci`/`nvidia-smi` calls

## Related Documentation

- [ENGINE_ARCHITECTURE.md](ENGINE_ARCHITECTURE.md) - Plugin system architecture
- [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) - Creating AI/ML plugins
- [requirements_photoholmes.txt](requirements_photoholmes.txt) - Photoholmes installation details
- [ai_detection/README.md](ai_detection/README.md) - AI detection module

## Components Using Detection

### Currently Implemented

- **Photoholmes Detection** (`plugins/ai_ml/photoholmes_detection.py`)
  - CPU methods: DQ, ZERO, Noisesniffer
  - GPU methods: Available if CUDA + weights downloaded
  
- **AI Detection** (`ai_detection/`)
  - SPAI detector
  - SDXL-detector (HuggingFace)
  - Multi-backend support

### Future Candidates

- **OpenCV Service** (`opencv_service/`)
  - Currently CPU-only Docker container
  - Could use GPU for faster manipulation detection
  
- **Perceptual Hashing** (`plugins/static/perceptualimagehash.py`)
  - Currently CPU-only
  - Could benefit from GPU acceleration for large batches

## Testing

### Verify Detection Works

```bash
# Run detection
make detect-system

# Should show:
# - GPU hardware status
# - Driver installation status  
# - CUDA version (if available)
# - Recommended backend
# - PyTorch index URL
```

### Test Installation

```bash
# Install photoholmes with detection
make photoholmes-setup

# Verify correct torch version installed
.venv/bin/pip list | grep torch
# CPU system: torch 2.10.0+cpu (or similar)
# GPU system: torch 2.10.0+cu124 (or similar CUDA version)
```

### Simulate Different Configurations

To test behavior on different systems:

1. **Simulate no GPU**: Edit script to return `False` from `check_nvidia_hardware()`
2. **Simulate no driver**: Edit script to return `False` from `check_nvidia_driver()`
3. **Simulate specific CUDA**: Edit script to return hardcoded version from `get_cuda_version()`

Then run `make detect-system` to see output.

## Migration Notes

### From Old Approach

**Before** (hardcoded CPU):
```makefile
photoholmes-setup:
    $(PIP) install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

**After** (automatic detection):
```makefile
photoholmes-setup:
    @INDEX_URL=$$($(SYSTEM_PYTHON) scripts/detect_system.py --index-url); \
    $(PIP) install --index-url $$INDEX_URL torch torchvision
```

### Breaking Changes

None. The detection script:
- Defaults to CPU if anything uncertain
- Uses same index URLs as before
- Produces identical installation on non-GPU systems

### Backwards Compatibility

Old installations with CPU torch will continue working. To get GPU acceleration:

1. Install NVIDIA drivers
2. Run `pip uninstall torch torchvision`
3. Run `make photoholmes-setup` (will detect GPU and install CUDA torch)
