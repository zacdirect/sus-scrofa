"""
ManTra-Net Model Definition - Keras 2.10+ Compatible Version

Updated imports for TensorFlow 2.x / Keras 2.10+ compatibility
Original: https://github.com/ISICV/ManTraNet
"""
import os
import numpy as np
import tensorflow as tf
from keras.layers import Layer, Input, GlobalAveragePooling2D, Lambda, Dense
from keras.layers import ConvLSTM2D, Conv2D, AveragePooling2D, BatchNormalization
from keras.constraints import UnitNorm, NonNeg, Constraint
from keras.activations import softmax
from keras.models import Model, load_model
from keras.initializers import Constant
from keras import backend as K


# Map old names to new
unit_norm = UnitNorm
non_neg = NonNeg


#################################################################################
# Custom Constraint: BayarConstraint
#################################################################################
class BayarConstraint(Constraint):
    """
    Bayar constraint for first layer (Bayar convolution).
    Forces the filter weights to have specific properties for image forensics.
    """
    def __call__(self, w):
        # Center weight must be -1
        w = w * K.cast(K.greater_equal(w,0.), K.floatx())
        w = w - K.mean(w, axis=[0,1,2], keepdims=True)
        return w / (K.sum(K.abs(w), axis=[0,1,2], keepdims=True) + 1e-10)


#################################################################################
# Conv2DSymPadding - Symmetric Padding Convolution
#################################################################################
class Conv2DSymPadding(Conv2D):
    """
    Convolutional layer with symmetric padding (replicate padding).
    """
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same',
                 data_format=None, dilation_rate=(1, 1), activation=None,
                 use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(Conv2DSymPadding, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',  # We handle padding manually
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.sym_padding = padding

    def call(self, inputs):
        # Apply symmetric padding
        if self.sym_padding == 'same':
            kernel_size = self.kernel_size[0] if isinstance(self.kernel_size, (list, tuple)) else self.kernel_size
            pad_size = (kernel_size - 1) // 2
            if pad_size > 0:
                inputs = tf.pad(inputs, 
                               [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], 
                               mode='SYMMETRIC')
        return super(Conv2DSymPadding, self).call(inputs)

    def get_config(self):
        config = super(Conv2DSymPadding, self).get_config()
        config.update({'sym_padding': self.sym_padding})
        return config


#################################################################################
# BayarConv2D - Bayar Convolution Layer
#################################################################################
class BayarConv2D(Conv2D):
    """
    Bayar convolution layer for image forensics.
    First layer that learns manipulation-sensitive filters.
    """
    def __init__(self, filters, kernel_size=5, strides=(1, 1), padding='same',
                 data_format=None, activation=None, use_bias=False,
                 kernel_initializer='glorot_uniform', **kwargs):
        
        # Remove any constraint from kwargs to avoid conflicts
        kwargs.pop('kernel_constraint', None)
        
        super(BayarConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_constraint=BayarConstraint(),
            **kwargs
        )

    def get_config(self):
        config = super(BayarConv2D, self).get_config()
        return config


#################################################################################
# Model Loading
#################################################################################
def load_pretrain_model_by_index(model_index, model_dir):
    """
    Load pretrained ManTraNet model.
    
    Args:
        model_index: Model version (0-4), typically use 4 for best performance
        model_dir: Directory containing ManTraNet_Ptrain{index}.h5 files
        
    Returns:
        Loaded Keras model
    """
    model_path = os.path.join(model_dir, f'ManTraNet_Ptrain{model_index}.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load with custom objects
    custom_objects = {
        'Conv2DSymPadding': Conv2DSymPadding,
        'BayarConstraint': BayarConstraint,
        'BayarConv2D': BayarConv2D,
        'unit_norm': unit_norm,
        'non_neg': non_neg
    }
    
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    return model
