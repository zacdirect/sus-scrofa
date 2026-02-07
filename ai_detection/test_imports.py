#!/usr/bin/env python3
"""
Quick test to verify SPAI model architecture can be imported.
Run this from the ai_detection directory to check if all dependencies are available.
"""

import sys
from pathlib import Path

# Add ai_detection to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all SPAI model components can be imported."""
    print("Testing SPAI model imports...")
    
    try:
        print("\n1. Testing config import...")
        from spai.config import get_inference_config, SPAIConfig
        print("   ✓ Config imports successful")
        
        print("\n2. Testing data transforms import...")
        from spai.data.transforms import build_inference_transform
        print("   ✓ Data transforms import successful")
        
        print("\n3. Testing model utilities import...")
        from spai.models import utils, filters
        print("   ✓ Model utilities import successful")
        
        print("\n4. Testing vision transformer import...")
        from spai.models import vision_transformer
        print("   ✓ Vision Transformer import successful")
        
        print("\n5. Testing backbones import...")
        from spai.models import backbones
        print("   ✓ Backbones import successful")
        
        print("\n6. Testing SID models import...")
        from spai.models import sid
        print("   ✓ SID models import successful")
        
        print("\n7. Testing model builder import...")
        from spai.models.build import build_cls_model
        print("   ✓ Model builder import successful")
        
        print("\n8. Testing inference API import...")
        from spai.inference import SPAIDetector
        print("   ✓ Inference API import successful")
        
        print("\n9. Creating test config...")
        config = get_inference_config()
        print(f"   ✓ Config created: {config.MODEL.TYPE}/{config.MODEL.SID_APPROACH}")
        
        print("\n✅ All imports successful!")
        print("\nNext steps:")
        print("  1. Run 'make weights' to download model weights")
        print("  2. Run 'make verify' to test full inference pipeline")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you've installed dependencies: make install")
        print("  2. Check if you're in the ai_detection virtual environment")
        print("  3. Verify all model files were copied correctly")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
