# AI Detection Setup Guide - SPAI

This document explains how to set up the **SPAI (Spectral AI-Generated Image Detector)** for Ghiro's AI detection plugin.

## About SPAI

**SPAI** is a state-of-the-art AI-generated image detector published at **CVPR 2025**.

- **Paper**: "Any-Resolution AI-Generated Image Detection by Spectral Learning"
- **Authors**: Karageorgiou et al. (CERTH/University of Amsterdam)
- **GitHub**: https://github.com/mever-team/spai
- **License**: Apache 2.0 (compatible with Ghiro)

### Key Features

- **Any-resolution detection**: Works on images of any size (no resizing needed)
- **Spectral learning**: Uses frequency domain analysis for robust detection
- **Latest generators**: Trained on Stable Diffusion 3, Midjourney v6.1, DALL-E, etc.
- **High accuracy**: State-of-the-art performance (CVPR 2025)

---

## Prerequisites

- **Python 3.11+** (SPAI requires Python 3.11 or newer)
- **PyTorch 2.0+** with torchvision
- **CUDA** (optional, for GPU acceleration)

---

## Installation Steps

### 1. Clone and Install SPAI

SPAI is a research repository that needs to be cloned and set up manually:

```bash
# Clone SPAI repository
cd /tmp
git clone https://github.com/mever-team/spai.git
cd spai

# Create Python 3.11+ environment (SPAI requires 3.11+)
conda create -n spai python=3.11
conda activate spai

# Install PyTorch (choose ONE):

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.4:
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# Install SPAI dependencies
pip install -r requirements.txt
```

**Important**: SPAI is not pip-installable. You must clone the repository and run it from that directory.

### 2. Download Model Weights

Download the pre-trained SPAI model weights (~100MB):

**Option 1: Google Drive (Official)**

1. Visit: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view
2. Download `spai.pth` or the weights file
3. Place in Ghiro's models directory: `models/weights/spai.pth`

**Option 2: Using gdown (Command Line)**

```bash
# Install gdown
pip install gdown

# Download weights
cd /tmp
gdown 1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI

# Move to Ghiro models directory
mv spai.pth /path/to/ghiro/models/weights/spai.pth
```

### 3. Verify Installation

Test that SPAI is working:

```bash
# Navigate to SPAI directory
cd /path/to/spai

# Activate your Python environment
conda activate spai

# Test SPAI command
python -m spai --help

# Should show SPAI help text with available commands
```

### 4. Set PYTHONPATH for Ghiro Integration

Since SPAI isn't installed as a package, add it to Python path:

```bash
# Option 1: Set environment variable
export PYTHONPATH="/path/to/spai:$PYTHONPATH"

# Option 2: Add to Ghiro's virtualenv activation script
echo 'export PYTHONPATH="/path/to/spai:$PYTHONPATH"' >> ~/venv/bin/activate

# Option 3: Add to ~/.bashrc for permanent setup
echo 'export PYTHONPATH="/path/to/spai:$PYTHONPATH"' >> ~/.bashrc
```

### 5. Configure Ghiro

The plugin will automatically detect SPAI if:
- SPAI directory is in PYTHONPATH or Python can find the `spai` module
- Model weights exist at `models/weights/spai.pth`
- The `python -m spai` command works

No additional Ghiro configuration needed!

---

## Quick Start

**Complete setup in 7 commands:**

```bash
# 1. Clone SPAI
cd /tmp
git clone https://github.com/mever-team/spai.git
cd spai

# 2. Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download weights
pip install gdown && gdown 1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI

# 5. Place weights in Ghiro's models directory
mkdir -p /path/to/ghiro/models/weights
mv spai.pth /path/to/ghiro/models/weights/

# 6. Add SPAI to Python path
export PYTHONPATH="/tmp/spai:$PYTHONPATH"

# 7. Test
python -m spai --help
```

---

## Configuration

### Default Config

SPAI uses `configs/spai.yaml` from the SPAI package by default. No changes needed for basic usage.

### Custom Config (Optional)

To use a custom configuration:

1. Copy SPAI's config: `cp /path/to/spai/configs/spai.yaml models/spai_config.yaml`
2. Edit as needed
3. Ghiro will automatically use `models/spai_config.yaml` if it exists

### GPU vs CPU

SPAI automatically detects and uses GPU if available. To force CPU:

```bash
# Set environment variable
export CUDA_VISIBLE_DEVICES=""
```

---

## Performance

### Inference Speed

| Hardware | Time per Image |
|----------|---------------|
| CPU (i7) | 5-10 seconds |
| GPU (RTX 3090) | 0.5-1 second |
| GPU (A100) | 0.3-0.5 seconds |

### Memory Requirements

- **CPU**: ~4GB RAM
- **GPU**: ~2GB VRAM
- **Model size**: ~100MB (ViT-B/16 based)

---

## Troubleshooting

### "SPAI module not available"

**Problem**: Plugin can't find SPAI.

**Solution**:
```bash
# Check if SPAI directory is accessible
ls /path/to/spai/spai/__main__.py

# Add to Python path
export PYTHONPATH="/path/to/spai:$PYTHONPATH"

# Test if Python can find it
python -c "import sys; print('\n'.join(sys.path))"
python -m spai --help
```

### "Model weights not found"

**Problem**: Weights file missing or in wrong location.

**Solution**:
```bash
# Check weights location
ls -lh models/weights/spai.pth

# If missing, download:
gdown 1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI
mv spai.pth models/weights/
```

### "Analysis timed out"

**Problem**: Inference taking too long (>60s).

**Solutions**:
- Use GPU instead of CPU
- Check system resources (CPU/memory)
- Reduce image size before analysis

### PyTorch Installation Issues

**For CPU-only systems:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Python Version Issues

SPAI requires Python 3.11+. Check your version:

```bash
python --version
# Should show Python 3.11 or newer

# If older, create new environment:
conda create -n ghiro python=3.11
conda activate ghiro
```

---

## Production Deployment

### Docker

Add to your Dockerfile:

```dockerfile
# Clone SPAI
RUN git clone https://github.com/mever-team/spai.git /opt/spai

# Install PyTorch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install SPAI dependencies
WORKDIR /opt/spai
RUN pip install -r requirements.txt

# Add SPAI to Python path
ENV PYTHONPATH="/opt/spai:${PYTHONPATH}"

# Copy model weights (download separately)
COPY models/weights/spai.pth /app/models/weights/spai.pth

WORKDIR /app
```

### Systemd Service

Ensure the service user can access model weights:

```bash
sudo mkdir -p /opt/ghiro/models/weights
sudo cp spai.pth /opt/ghiro/models/weights/
sudo chown -R ghiro:ghiro /opt/ghiro/models
```

### Hosting Model Weights Internally

For production, host weights on your infrastructure:

```bash
# Option 1: S3/CDN
aws s3 cp spai.pth s3://your-bucket/models/spai.pth

# Option 2: Internal server
scp spai.pth user@internal-server:/var/www/models/
```

Then update the download instructions for your team.

---

## Alternative: Manual Inference (Advanced)

If you prefer to run SPAI manually:

```bash
# Place images in input directory
mkdir -p /tmp/input
cp image.jpg /tmp/input/

# Run inference
python -m spai infer \
  --input /tmp/input \
  --output /tmp/output \
  --model models/weights/spai.pth \
  --cfg configs/spai.yaml \
  --batch-size 1

# Check results
cat /tmp/output/*.csv
```

---

## Comparison with Previous Detector

| Feature | SPAI (New) | GRIP-UNINA (Old) |
|---------|-----------|------------------|
| **Year** | 2025 (CVPR) | 2022 (ICASSP) |
| **Resolution** | Any size | Fixed 224x224 |
| **Method** | Spectral learning | ResNet-50 CNN |
| **Training** | SD3, MJ v6.1, DALL-E | SD1.5, MJ v3, ProGAN |
| **Setup** | CLI tool | Manual PyTorch code |
| **Performance** | State-of-the-art | Good |

---

## References

- **Paper**: https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html
- **GitHub**: https://github.com/mever-team/spai
- **Weights**: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view

---

## Support

For issues with:
- **SPAI itself**: https://github.com/mever-team/spai/issues
- **Ghiro integration**: Open an issue on Ghiro's repository
- **Model downloads**: Contact d.karageorgiou@uva.nl

---

## License

- **SPAI**: Apache 2.0
- **Ghiro**: Ghiro license (see docs/LICENSE.txt)
- **Model weights**: Apache 2.0 (research use)

Both are compatible for commercial and research use.
