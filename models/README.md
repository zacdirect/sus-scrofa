# Models Directory

This directory contains AI detection model weights and related files.

## Structure

```
models/
├── weights/          # Pre-trained model weights (gitignored)
│   └── spai.pth     # SPAI model checkpoint (~100MB)
└── README.md        # This file
```

## Setup

Model weights are **not included in the repository** due to their size (~100MB).

Download instructions: See `AI_DETECTION_SETUP.md` in the project root.

## What's Gitignored

- `models/weights/` - All model weight files
- `models/*.pth` - Any checkpoint files
- `models/*.onnx` - Exported ONNX models
