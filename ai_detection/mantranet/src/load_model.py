"""
ManTraNet Model Loader - TensorFlow.keras Compatibility Layer

This module handles loading ManTraNet models using TensorFlow's bundled keras (tf.keras).
ManTraNet was written for Keras 2.x. We map tf.keras to the keras namespace.
"""

import os
import sys

# Use TensorFlow's bundled Keras 2 API
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf

# Create keras namespace mappings for modelCore imports
sys.modules['keras'] = tf.keras
sys.modules['keras.layers'] = tf.keras.layers
sys.modules['keras.models'] = tf.keras.models
sys.modules['keras.constraints'] = tf.keras.constraints
sys.modules['keras.activations'] = tf.keras.activations
sys.modules['keras.initializers'] = tf.keras.initializers
sys.modules['keras.engine'] = tf.keras.engine
sys.modules['keras.backend'] = tf.keras.backend

# Handle legacy imports that don't exist in modern TensorFlow
class LegacyInterfaces:
    """Stub for keras.legacy.interfaces which doesn't exist in tf.keras."""
    @staticmethod
    def legacy_conv2d_support(func):
        return func

class LegacyModule:
    """Stub module for keras.legacy."""
    interfaces = LegacyInterfaces()

sys.modules['keras.legacy'] = LegacyModule()

# Handle keras.layers.convolutional which is reorganized in tf.keras
class ConvolutionalLayers:
    """Adapter for keras.layers.convolutional._Conv."""
    _Conv = tf.keras.layers.Conv2D.__bases__[0]  # Get base class

sys.modules['keras.layers.convolutional'] =  ConvolutionalLayers()


def load_mantranet_model(model_index, model_dir):
    """
    Load ManTraNet pretrained model using original modelCore with tf.keras.
    
    Args:
        model_index: Model version (0-4), typically 4 for best performance
        model_dir: Directory containing ManTraNet_Ptrain{index}.h5 files
        
    Returns:
        Loaded Keras model ready for inference
    """
    # Import original modelCore now that keras namespace is properly mapped
    try:
        import modelCore
    except ImportError as e:
        raise ImportError(
            f"Failed to import modelCore: {e}\n"
            "Make sure ai_detection/mantranet/src/modelCore.py exists and is accessible."
        )
    
    # Use original implementation
    model = modelCore.load_pretrain_model_by_index(model_index, model_dir)
    return model
