"""
SPAI Configuration

Minimal configuration for inference-only usage.
Based on SPAI's config.py but simplified for embedded use.
"""

from yacs.config import CfgNode as CN


def get_inference_config():
    """
    Get default SPAI inference configuration.
    
    Returns minimal config needed for loading model and running inference.
    """
    config = CN()
    
    # Model architecture
    config.MODEL = CN()
    config.MODEL.TYPE = 'vit'  # Vision Transformer
    config.MODEL.NAME = 'spai'
    config.MODEL.NUM_CLASSES = 2  # Binary classification (real=0 vs AI-generated=1)
    config.MODEL.DROP_RATE = 0.0
    config.MODEL.SID_DROPOUT = 0.5
    config.MODEL.DROP_PATH_RATE = 0.1
    config.MODEL.SID_APPROACH = "freq_restoration"  # Frequency restoration approach
    config.MODEL.RESOLUTION_MODE = "arbitrary"  # Supports any resolution
    config.MODEL.FEATURE_EXTRACTION_BATCH = 400
    config.MODEL_WEIGHTS = "mfm"  # Pre-trained on MFM
    
    # Vision Transformer specific
    config.MODEL.VIT = CN()
    config.MODEL.VIT.PATCH_SIZE = 16
    config.MODEL.VIT.IN_CHANS = 3
    config.MODEL.VIT.EMBED_DIM = 768
    config.MODEL.VIT.DEPTH = 12
    config.MODEL.VIT.NUM_HEADS = 12
    config.MODEL.VIT.MLP_RATIO = 4
    config.MODEL.VIT.QKV_BIAS = True
    config.MODEL.VIT.USE_APE = True  # Absolute position embedding
    config.MODEL.VIT.USE_FPE = False
    config.MODEL.VIT.USE_RPB = False
    config.MODEL.VIT.USE_SHARED_RPB = False
    config.MODEL.VIT.USE_MEAN_POOLING = True  # Official config uses mean pooling
    config.MODEL.VIT.INIT_VALUES = 0.1
    config.MODEL.VIT.USE_INTERMEDIATE_LAYERS = True
    config.MODEL.VIT.INTERMEDIATE_LAYERS = tuple(range(12))
    config.MODEL.VIT.RETURN_FEATURES = True
    config.MODEL.VIT.FEATURES_PROCESSOR = "fre"  # Use frequency restoration estimator
    config.MODEL.VIT.PROJECTION_DIM = 1024  # Match FRE.PROJ_DIM
    config.MODEL.VIT.PROJECTION_LAYERS = 2  # Match FRE.PROJ_LAYERS
    config.MODEL.VIT.PATCH_PROJECTION = True
    config.MODEL.VIT.PATCH_PROJECTION_PER_FEATURE = True
    config.MODEL.VIT.PATCH_POOLING = False
    
    # Patch-based model for arbitrary resolution
    config.MODEL.PATCH_VIT = CN()
    config.MODEL.PATCH_VIT.IMG_PATCH_SIZE = 224
    config.MODEL.PATCH_VIT.IMG_PATCH_STRIDE = 224
    config.MODEL.PATCH_VIT.MINIMUM_PATCHES = 4  # From official config (not 1)
    config.MODEL.PATCH_VIT.CLS_VECTOR_DIM = 1096  # From checkpoint dimensions (not 6144)
    config.MODEL.PATCH_VIT.NUM_HEADS = 12  # Match checkpoint
    config.MODEL.PATCH_VIT.ATTN_EMBED_DIM = 1536
    config.MODEL.PATCH_VIT.DROPOUT = 0.2
    
    # Frequency restoration estimator
    config.MODEL.FRE = CN()
    config.MODEL.FRE.FEATURES_NUM = 12
    config.MODEL.FRE.PROJ_DIM = 1024
    config.MODEL.FRE.PROJ_LAYERS = 2
    config.MODEL.FRE.PATCH_PROJECTION = True
    config.MODEL.FRE.PATCH_PROJECTION_PER_FEATURE = True
    config.MODEL.FRE.PROJ_LAST_LAYER_ACTIVATION_TYPE = None
    config.MODEL.FRE.PROJECTOR_LAST_LAYER_ACTIVATION_TYPE = None  # No activation on last layer (official config)
    config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH = True  # From checkpoint - adds original image features
    config.MODEL.FRE.DROPOUT = 0.5
    config.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY = False
    config.MODEL.FRE.MASKING_RADIUS = 16  # Frequency masking radius for FRE
    
    # Classification head settings
    config.MODEL.CLS_HEAD = CN()
    config.MODEL.CLS_HEAD.MLP_RATIO = 3  # From checkpoint (not 4)
    
    # SID model settings
    config.MODEL.SID_DROPOUT = 0.1  # Dropout for SID model
    
    # Patch-based ViT settings (for arbitrary resolution mode)
    config.MODEL.PATCH_VIT = CN()
    config.MODEL.PATCH_VIT.PATCH_STRIDE = 112  # Stride for patch extraction
    config.MODEL.PATCH_VIT.ATTN_EMBED_DIM = 512  # Attention embedding dimension
    config.MODEL.PATCH_VIT.NUM_HEADS = 8  # Number of attention heads
    config.MODEL.PATCH_VIT.MINIMUM_PATCHES = 4  # Minimum patches for processing
    
    # Data processing
    config.DATA = CN()
    config.DATA.IMG_SIZE = 224
    config.DATA.INTERPOLATION = 'bicubic'
    config.DATA.MASK_RADIUS_1 = 16  # Frequency masking radius
    config.DATA.SAMPLE_RATIO = 0.5
    
    # Inference settings
    config.TEST = CN()
    config.TEST.CROP = True
    config.TEST.MAX_SIZE = None
    config.TEST.ORIGINAL_RESOLUTION = True  # Use original resolution (official config)
    
    # Training settings (needed for model building even in inference mode)
    config.TRAIN = CN()
    config.TRAIN.MODE = "supervised"  # Enables classification head for inference
    
    # Patch-based ViT settings (for arbitrary resolution mode)
    # These control how large images are divided into patches for analysis
    config.MODEL.PATCH_VIT = CN()
    config.MODEL.PATCH_VIT.PATCH_STRIDE = 224  # Stride between patches (same as IMG_SIZE = non-overlapping)
    config.MODEL.PATCH_VIT.ATTN_EMBED_DIM = 1536  # Attention embedding dimension (from checkpoint)
    config.MODEL.PATCH_VIT.NUM_HEADS = 12  # Number of attention heads (from checkpoint)
    config.MODEL.PATCH_VIT.MINIMUM_PATCHES = 1  # Minimum patches to process (from checkpoint)
    
    # Device
    config.DEVICE = "cuda"  # Will be overridden by user
    
    config.freeze()
    return config


class SPAIConfig:
    """
    SPAI configuration wrapper for easy customization.
    """
    
    def __init__(self, device="cuda", batch_size=1):
        """
        Initialize SPAI configuration.
        
        Args:
            device: "cuda" or "cpu"
            batch_size: Batch size for inference (usually 1 for single image)
        """
        self.config = get_inference_config()
        
        # Allow mutation for user overrides
        self.config.defrost()
        self.config.DEVICE = device
        self.config.DATA.BATCH_SIZE = batch_size
        self.config.freeze()
    
    def get(self):
        """Get the underlying YACS config object."""
        return self.config
