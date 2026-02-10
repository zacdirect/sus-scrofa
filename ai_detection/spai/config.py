"""
SPAI Configuration

Minimal configuration for inference-only usage.
Matches the official spai.yaml from the SPAI CVPR2025 release.

Reference: https://github.com/o6-webwork/deepfake-detection/blob/main/spai/configs/spai.yaml
"""

from yacs.config import CfgNode as CN


def get_inference_config():
    """
    Get default SPAI inference configuration.

    Returns minimal config needed for loading model and running inference.
    All values match the official spai.yaml and upstream config.py defaults.
    """
    config = CN()

    # -------------------------------------------------------------------------
    # Model architecture — matches spai.yaml
    # -------------------------------------------------------------------------
    config.MODEL = CN()
    config.MODEL.TYPE = 'vit'
    config.MODEL.NAME = 'finetune'
    config.MODEL.NUM_CLASSES = 2
    config.MODEL.DROP_RATE = 0.0
    config.MODEL.DROP_PATH_RATE = 0.1
    config.MODEL.SID_APPROACH = "freq_restoration"
    config.MODEL.RESOLUTION_MODE = "arbitrary"
    config.MODEL.FEATURE_EXTRACTION_BATCH = 400
    # Normalization the model expects on its input tensor.
    # "positive_0_1" means pixels in [0, 1] — the model normalizes internally
    # after FFT frequency decomposition.
    config.MODEL.REQUIRED_NORMALIZATION = "positive_0_1"
    # Dropout for trainable SID layers (FRE projectors, cls_head, patch attention).
    # Upstream default is 0.5; spai.yaml does not override it.
    config.MODEL.SID_DROPOUT = 0.5

    # Top-level: backbone weight type
    config.MODEL_WEIGHTS = "mfm"

    # -------------------------------------------------------------------------
    # Vision Transformer — matches spai.yaml MODEL.VIT section
    # -------------------------------------------------------------------------
    config.MODEL.VIT = CN()
    config.MODEL.VIT.PATCH_SIZE = 16
    config.MODEL.VIT.IN_CHANS = 3
    config.MODEL.VIT.EMBED_DIM = 768
    config.MODEL.VIT.DEPTH = 12
    config.MODEL.VIT.NUM_HEADS = 12
    config.MODEL.VIT.MLP_RATIO = 4
    config.MODEL.VIT.QKV_BIAS = True
    # INIT_VALUES: None means no LayerScale (no gamma_1/gamma_2 params).
    # The checkpoint was trained without LayerScale.
    config.MODEL.VIT.INIT_VALUES = None
    config.MODEL.VIT.USE_APE = True       # Learnable absolute positional embedding
    config.MODEL.VIT.USE_FPE = False      # No fixed positional embedding
    config.MODEL.VIT.USE_RPB = False      # No relative position bias
    config.MODEL.VIT.USE_SHARED_RPB = False
    config.MODEL.VIT.USE_MEAN_POOLING = True
    config.MODEL.VIT.USE_INTERMEDIATE_LAYERS = True
    config.MODEL.VIT.INTERMEDIATE_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    config.MODEL.VIT.PROJECTION_DIM = 1024
    config.MODEL.VIT.PROJECTION_LAYERS = 2
    config.MODEL.VIT.PATCH_PROJECTION = True
    config.MODEL.VIT.PATCH_PROJECTION_PER_FEATURE = True
    config.MODEL.VIT.PATCH_POOLING = "mean"

    # -------------------------------------------------------------------------
    # Frequency Restoration Estimator — matches spai.yaml MODEL.FRE section
    # -------------------------------------------------------------------------
    config.MODEL.FRE = CN()
    config.MODEL.FRE.MASKING_RADIUS = 16
    config.MODEL.FRE.PROJECTOR_LAST_LAYER_ACTIVATION_TYPE = None
    config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH = True
    config.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY = False

    # -------------------------------------------------------------------------
    # Classification head — matches spai.yaml MODEL.CLS_HEAD section
    # -------------------------------------------------------------------------
    config.MODEL.CLS_HEAD = CN()
    config.MODEL.CLS_HEAD.MLP_RATIO = 3

    # -------------------------------------------------------------------------
    # PatchBasedMFViT (arbitrary resolution) — matches spai.yaml MODEL.PATCH_VIT
    # Defined ONCE. Official values from spai.yaml.
    # -------------------------------------------------------------------------
    config.MODEL.PATCH_VIT = CN()
    config.MODEL.PATCH_VIT.PATCH_STRIDE = 224
    config.MODEL.PATCH_VIT.NUM_HEADS = 12
    config.MODEL.PATCH_VIT.ATTN_EMBED_DIM = 1536
    config.MODEL.PATCH_VIT.MINIMUM_PATCHES = 4

    # -------------------------------------------------------------------------
    # Data processing
    # -------------------------------------------------------------------------
    config.DATA = CN()
    config.DATA.IMG_SIZE = 224
    config.DATA.INTERPOLATION = 'bicubic'

    # -------------------------------------------------------------------------
    # Test / Inference settings — matches spai.yaml TEST section
    # -------------------------------------------------------------------------
    config.TEST = CN()
    config.TEST.CROP = True
    config.TEST.MAX_SIZE = None
    # ORIGINAL_RESOLUTION=True: keep the original image size, pad to >= 224.
    # The PatchBasedMFViT model handles patchification internally.
    config.TEST.ORIGINAL_RESOLUTION = True

    # -------------------------------------------------------------------------
    # Training settings (needed by build_mf_vit even for inference)
    # -------------------------------------------------------------------------
    config.TRAIN = CN()
    config.TRAIN.MODE = "supervised"

    # -------------------------------------------------------------------------
    # Device (overridden by user)
    # -------------------------------------------------------------------------
    config.DEVICE = "cuda"

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
