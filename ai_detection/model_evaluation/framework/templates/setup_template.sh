#!/bin/bash
# Setup script for AI detection model
#
# This script should:
# 1. Download model weights
# 2. Install dependencies
# 3. Verify installation

set -e  # Exit on error

echo "================================================"
echo "Setting up AI Detection Model"
echo "================================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create weights directory
echo ""
echo "Creating directories..."
mkdir -p weights
mkdir -p tests

# Download model weights
echo ""
echo "Downloading model weights..."
# TODO: Add weight download command
# Example:
# wget -O weights/model.pth "https://example.com/model_weights.pth"
# Or:
# curl -L -o weights/model.pth "https://example.com/model_weights.pth"

echo "⚠ Weight download not configured"
echo "  Please manually download weights to: weights/"
echo "  Or update this script with download URL"

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠ No requirements.txt found"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import sys
sys.path.insert(0, '../../..')

try:
    from detector import MyDetector
    detector = MyDetector()
    if detector.check_deps():
        print('✓ Dependencies verified')
    else:
        print('✗ Dependencies check failed')
        sys.exit(1)
except Exception as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"

echo ""
echo "================================================"
echo "✓ Setup complete"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Edit detector.py to implement your model"
echo "  2. Test: python detector.py <test_image.jpg>"
echo "  3. Evaluate: make test-candidate NAME=$(basename $SCRIPT_DIR)"
