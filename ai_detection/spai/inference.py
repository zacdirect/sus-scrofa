"""
SPAI Inference Interface

High-level API for AI-generated image detection using SPAI.
"""

import os
import pathlib
import logging
from typing import Union, List, Dict, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .config import SPAIConfig
from .models.build import build_cls_model
from .data.transforms import build_inference_transform


logger = logging.getLogger(__name__)


class SPAIDetector:
    """
    SPAI AI-Generated Image Detector.
    
    This class provides a simple interface for detecting AI-generated images
    using the SPAI (Spectral AI-Generated Image Detector) method.
    
    Example:
        >>> detector = SPAIDetector("weights/spai.pth", device="cuda")
        >>> result = detector.predict("image.jpg")
        >>> print(f"AI-generated probability: {result['score']:.2f}")
    """
    
    def __init__(
        self,
        weights_path: Union[str, pathlib.Path],
        device: str = "cuda",
        config: Optional[SPAIConfig] = None
    ):
        """
        Initialize SPAI detector.
        
        Args:
            weights_path: Path to SPAI model weights (.pth file)
            device: "cuda" or "cpu"
            config: Optional custom SPAIConfig. If None, uses defaults.
        
        Raises:
            FileNotFoundError: If weights file doesn't exist
            RuntimeError: If CUDA requested but not available
        """
        self.weights_path = pathlib.Path(weights_path)
        
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {self.weights_path}\n"
                f"Download from: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view"
            )
        
        # Setup device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)
        
        # Load configuration
        if config is None:
            config = SPAIConfig(device=device)
        self.config = config.get()
        
        # Load model
        logger.info(f"Loading SPAI model from {self.weights_path}")
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Setup image preprocessing
        self.transform = build_inference_transform(self.config)
        
        logger.info(f"SPAI detector initialized on {device}")
    
    def _load_model(self):
        """Load SPAI model from checkpoint."""
        try:
            # Build model architecture
            model = build_cls_model(self.config)
            
            # Load weights
            checkpoint = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=False
            )
            
            # Extract model state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'encoder.' prefix if present (from pre-training)
            if any(k.startswith('encoder.') for k in state_dict.keys()):
                state_dict = {
                    k.replace('encoder.', ''): v
                    for k, v in state_dict.items()
                    if k.startswith('encoder.')
                }
            
            # Load into model
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            if missing:
                logger.warning(f"Missing keys in checkpoint: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SPAI model: {e}")
    
    def predict(
        self,
        image: Union[str, pathlib.Path, np.ndarray, Image.Image],
        return_details: bool = False
    ) -> Dict:
        """
        Predict if an image is AI-generated.
        
        Args:
            image: Path to image file, numpy array, or PIL Image
            return_details: If True, return additional details (attention maps, etc.)
        
        Returns:
            Dictionary with:
                - score: AI-generated probability (0.0-1.0)
                - confidence: Model confidence in prediction
                - is_ai_generated: Boolean classification (threshold 0.5)
                - logit: Raw model logit value
        
        Example:
            >>> result = detector.predict("test.jpg")
            >>> if result['is_ai_generated']:
            ...     print(f"Detected AI-generated with {result['score']:.1%} confidence")
        """
        # Load and preprocess image
        img_tensor = self._preprocess_image(image)
        img_tensor = img_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            if isinstance(img_tensor, list):
                # Arbitrary resolution mode
                output = self.model(img_tensor, self.config.MODEL.FEATURE_EXTRACTION_BATCH)
            else:
                # Fixed resolution mode
                output = self.model(img_tensor)
            
            # Get probability score
            probability = torch.sigmoid(output).item()
            logit = output.item()
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(probability - 0.5) * 2.0
        
        result = {
            'score': float(probability),
            'confidence': float(confidence),
            'is_ai_generated': bool(probability > 0.5),
            'logit': float(logit),
            'enabled': True
        }
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, pathlib.Path, np.ndarray, Image.Image]],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Predict multiple images in batches.
        
        Args:
            images: List of images (paths, arrays, or PIL Images)
            batch_size: Number of images to process at once
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Process each image in batch
            for img in batch:
                result = self.predict(img)
                results.append(result)
        
        return results
    
    def _preprocess_image(
        self,
        image: Union[str, pathlib.Path, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image in various formats
        
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image if path
        if isinstance(image, (str, pathlib.Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Apply transformations
        img_tensor = self.transform(image)
        
        # Add batch dimension if needed
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    @staticmethod
    def check_availability() -> Dict[str, bool]:
        """
        Check if SPAI is available and properly configured.
        
        Returns:
            Dictionary with availability status
        """
        status = {
            'pytorch_available': False,
            'cuda_available': False,
            'spai_available': False,
            'weights_available': False
        }
        
        try:
            import torch
            status['pytorch_available'] = True
            status['cuda_available'] = torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            from .models.build import build_cls_model
            status['spai_available'] = True
        except ImportError:
            pass
        
        # Check for weights file
        weights_path = pathlib.Path(__file__).parent.parent / "weights" / "spai.pth"
        status['weights_available'] = weights_path.exists()
        
        return status


def quick_predict(
    image_path: Union[str, pathlib.Path],
    weights_path: Union[str, pathlib.Path],
    device: str = "cuda"
) -> Dict:
    """
    Quick one-shot prediction without persistent detector.
    
    Useful for testing, but inefficient for multiple images
    (loads model each time).
    
    Args:
        image_path: Path to image
        weights_path: Path to SPAI weights
        device: "cuda" or "cpu"
    
    Returns:
        Prediction dictionary
    """
    detector = SPAIDetector(weights_path, device=device)
    return detector.predict(image_path)
