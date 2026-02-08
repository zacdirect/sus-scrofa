# sdxl_detector — Organika/sdxl-detector

## Model Information
- **Source**: https://huggingface.co/Organika/sdxl-detector
- **Base model**: https://huggingface.co/umm-maybe/AI-image-detector
- **Architecture**: SwinForImageClassification (Swin Transformer Large, 86.8M params)
- **Input**: 224×224 RGB images
- **Output**: Binary classification — `artificial` (AI) vs `human` (real)
- **License**: CC-BY-NC-3.0 (non-commercial)
- **Downloads**: ~23K/month on HuggingFace

## Published Metrics
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 98.1%  |
| F1        | 97.3%  |
| Precision | 99.5%  |
| Recall    | 95.3%  |
| AUC       | 99.8%  |

## Why This Model?
- Fine-tuned specifically for modern diffusion model outputs (SDXL)
- Broader subject range than predecessor (Wikimedia training data vs Reddit art)
- Standard HuggingFace pipeline — no custom preprocessing needed
- Active community (65+ Spaces, 6 fine-tunes)
- Weights auto-download from HuggingFace Hub

## Setup
```bash
pip install -r requirements.txt
# Weights auto-download on first run (~350MB)
```

## Quick Test
```bash
python detector.py /path/to/image.jpg
```

## Notes
- First run downloads the model (~350MB) — subsequent runs use cache
- CPU inference takes ~1-3s per image; GPU is faster
- Model may underperform on images from older generators (VQGAN+CLIP)
