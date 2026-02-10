"""
ManTraNet Model - Modern TensorFlow 2.20 / Keras 3.x Implementation

Rebuilt from original ManTraNet architecture for compatibility with modern TensorFlow.
Original: https://github.com/ISICV/ManTraNet (2019, Keras 2.2.x)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, constraints, backend as K


#################################################################################
# Custom Constraints
#################################################################################

class BayarConstraint(constraints.Constraint):
    """
    Bayar constraint for forensic conv layer.
    Forces center weight to be -1 and normalizes other weights.
    """
    def __init__(self):
        super().__init__()
        self.mask = None
    
    def __call__(self, w):
        if self.mask is None:
            # Initialize mask on first call
            nb_rows, nb_cols, nb_inputs, nb_outputs = K.int_shape(w)
            m = np.zeros([nb_rows, nb_cols, nb_inputs, nb_outputs], dtype='float32')
            m[nb_rows//2, nb_cols//2] = 1.0
            self.mask = tf.constant(m, dtype=w.dtype)
        
        # Apply constraint: zero center, normalize, subtract 1 from center
        w = w * (1 - self.mask)
        rest_sum = tf.reduce_sum(w, axis=(0, 1), keepdims=True)
        w = w / (rest_sum + K.epsilon())
        w = w - self.mask
        return w


class UnitNormConstraint(constraints.Constraint):
    """Unit norm constraint for a specific axis."""
    def __init__(self, axis=-2):
        super().__init__()
        self.axis = axis
    
    def __call__(self, w):
        return w / (K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True)) + K.epsilon())


#################################################################################
# Custom Layers
#################################################################################

class Conv2DSymPadding(layers.Layer):
    """
    2D convolution with symmetric padding (replication padding).
    """
    def __init__(self, filters, kernel_size, strides=(1, 1), activation=None,
                 use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint) if kernel_constraint else None
        self.bias_constraint = keras.constraints.get(bias_constraint) if bias_constraint else None
        
    def build(self, input_shape):
        input_channels = input_shape[-1]
        kernel_shape = self.kernel_size + (input_channels, self.filters)
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint,
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                constraint=self.bias_constraint,
                trainable=True
            )
        else:
            self.bias = None
        
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply symmetric padding
        ph, pw = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        if ph > 0 or pw > 0:
            inputs = tf.pad(inputs, [[0, 0], [ph, ph], [pw, pw], [0, 0]], mode='SYMMETRIC')
        
        # Convolve
        outputs = K.conv2d(inputs, self.kernel, strides=self.strides, padding='valid')
        
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias)
        
        if self.activation is not None:
            outputs = self.activation(outputs)
        
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
        })
        return config


class CombinedConv2D(Conv2DSymPadding):
    """
    Combined convolution: regular kernels + SRM kernels + Bayar constrained kernels.
    Specific to forensic image analysis (ManTraNet first layer).
    """
    def __init__(self, filters, kernel_size=(5, 5), **kwargs):
        kwargs['use_bias'] = False  # Combined conv doesn't use bias
        super().__init__(filters, kernel_size, **kwargs)
    
    def _get_srm_kernels(self):
        """Build SRM (Spatial Rich Model) kernels for forensics."""
        # SRM kernel 1
        srm1 = np.zeros([5, 5], dtype='float32')
        srm1[1:-1, 1:-1] = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype='float32')
        srm1 /= 4.0
        
        # SRM kernel 2
        srm2 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype='float32') / 12.0
        
        # SRM kernel 3
        srm3 = np.zeros([5, 5], dtype='float32')
        srm3[2, 1:-1] = np.array([1, -2, 1], dtype='float32')
        srm3 /= 2.0
        
        # Stack for all RGB channels
        kernel_list = []
        for srm in [srm1, srm2, srm3]:
            for ch in range(3):
                ch_kernel = np.zeros([5, 5, 3], dtype='float32')
                ch_kernel[:, :, ch] = srm
                kernel_list.append(ch_kernel)
        
        return np.stack(kernel_list, axis=-1)  # Shape: (5, 5, 3, 9)
    
    def build(self, input_shape):
        input_channels = input_shape[-1]
        
        # 1. Regular conv kernels (trainable)
        regular_filters = self.filters - 9 - 3  # 9 SRM + 3 Bayar
        if regular_filters >= 1:
            self.regular_kernel = self.add_weight(
                name='regular_kernel',
                shape=self.kernel_size + (input_channels, regular_filters),
                initializer=self.kernel_initializer,
                trainable=True
            )
        else:
            self.regular_kernel = None
        
        # 2. SRM kernels (not trainable)
        srm_kernels = self._get_srm_kernels()
        self.srm_kernel = tf.constant(srm_kernels, dtype=tf.float32)
        
        # 3. Bayar kernels (trainable with constraint)
        bayar_constraint = BayarConstraint()
        self.bayar_kernel = self.add_weight(
            name='bayar_kernel',
            shape=self.kernel_size + (input_channels, 3),
            initializer=self.kernel_initializer,
            constraint=bayar_constraint,
            trainable=True
        )
        
        # Note: we override parent's build so we must set built flag
        self.built = True
    
    def call(self, inputs):
        # Apply symmetric padding
        ph, pw = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        if ph > 0 or pw > 0:
            inputs = tf.pad(inputs, [[0, 0], [ph, ph], [pw, pw], [0, 0]], mode='SYMMETRIC')
        
        # Combine all kernels
        kernel_parts = [self.srm_kernel, self.bayar_kernel]
        if self.regular_kernel is not None:
            kernel_parts.insert(0, self.regular_kernel)
        
        combined_kernel = tf.concat(kernel_parts, axis=-1)
        
        # Convolve
        outputs = K.conv2d(inputs, combined_kernel, strides=self.strides, padding='valid')
        
        if self.activation is not None:
            outputs = self.activation(outputs)
        
        return outputs


class GlobalStd2D(layers.Layer):
    """Global standard deviation layer with learnable minimum."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.min_std = self.add_weight(
            name='min_std',
            shape=(1, 1, 1, num_channels),
            initializer=keras.initializers.Constant(1e-4),
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        # Calculate std per channel, per batch item, over spatial dimensions
        mean = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=[1, 2], keepdims=True)
        std = tf.sqrt(variance + K.epsilon())
        return tf.maximum(std, self.min_std)


class MantraNetConvLSTM(layers.Layer):
    """
    ConvLSTM2D implementation that matches ManTraNet's weight structure.
    Compatible with weights saved from Keras 2.x ConvLSTM2D.
    """
    def __init__(self, filters, kernel_size, activation='tanh',
                 recurrent_activation='hard_sigmoid', padding='same',
                 return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.activation = keras.activations.get(activation)
        self.recurrent_activation = keras.activations.get(recurrent_activation)
        self.padding = padding
        self.return_sequences = return_sequences
    
    def build(self, input_shape):
        # input_shape: (batch, time, height, width, channels)
        input_dim = input_shape[-1]
        
        # Kernel weights for input: (kh, kw, input_dim, filters * 4)
        # Combined for [input_gate, forget_gate, cell_gate, output_gate]
        self.kernel = self.add_weight(
            name='kernel',
            shape=self.kernel_size + (input_dim, self.filters * 4),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Recurrent kernel weights: (kh, kw, filters, filters * 4)
        self.recurrent_kernel = self.add_weight(
            name='recurrent_kernel', 
            shape=self.kernel_size + (self.filters, self.filters * 4),
            initializer='orthogonal',
            trainable=True
        )
        
        # Bias: (filters * 4,)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters * 4,),
            initializer='zeros',
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        # inputs: (batch, time, height, width, channels)
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[2]
        width = tf.shape(inputs)[3]
        time_steps = inputs.shape[1]  # Static dimension
        
        # Initialize states
        h = tf.zeros((batch_size, height, width, self.filters))
        c = tf.zeros((batch_size, height, width, self.filters))
        
        outputs = []
        
        # Manual unroll over time steps
        for t in range(time_steps):
            x_t = inputs[:, t, :, :, :]
            
            # Compute gates using combined kernels
            # Apply input convolution
            z_input = K.conv2d(x_t, self.kernel, padding=self.padding)
            
            # Apply recurrent convolution
            z_recurrent = K.conv2d(h, self.recurrent_kernel, padding=self.padding)
            
            # Combine and add bias
            z = z_input + z_recurrent
            z = K.bias_add(z, self.bias)
            
            # Split into 4 gates
            z_i, z_f, z_c, z_o = tf.split(z, 4, axis=-1)
            
            # Apply activations
            i = self.recurrent_activation(z_i)  # Input gate
            f = self.recurrent_activation(z_f)  # Forget gate
            c_new = f * c + i * self.activation(z_c)  # New cell state
            o = self.recurrent_activation(z_o)  # Output gate
            h_new = o * self.activation(c_new)  # New hidden state
            
            c = c_new
            h = h_new
            
            if self.return_sequences:
                outputs.append(h)
        
        if self.return_sequences:
            return tf.stack(outputs, axis=1)
        else:
            return h
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': keras.activations.serialize(self.activation),
            'recurrent_activation': keras.activations.serialize(self.recurrent_activation),
            'padding': self.padding,
            'return_sequences': self.return_sequences,
        })
        return config


class NestedWindowAverageFeatExtractor(layers.Layer):
    """Extract features using nested window averaging at multiple scales."""
    def __init__(self, window_size_list=[7, 15, 31], **kwargs):
        super().__init__(**kwargs)
        self.window_size_list = window_size_list
    
    def call(self, inputs):
        # Create temporal dimension by stacking different pooling scales
        pooled_list = []
        for window_size in self.window_size_list:
            pooled = layers.AveragePooling2D(pool_size=window_size, strides=1, padding='same')(inputs)
            deviation = pooled - inputs  # Deviation from average
            pooled_list.append(deviation)
        
        # Stack along new time dimension: (batch, time, height, width, channels)
        output = tf.stack(pooled_list, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.window_size_list), input_shape[1], input_shape[2], input_shape[3])
    
    def get_config(self):
        config = super().get_config()
        config['window_size_list'] = self.window_size_list
        return config
    """Extract features using nested window averaging at multiple scales."""
    def __init__(self, window_size_list=[7, 15, 31], **kwargs):
        super().__init__(**kwargs)
        self.window_size_list = window_size_list
    
    def call(self, inputs):
        # Create temporal dimension by stacking different pooling scales
        pooled_list = []
        for window_size in self.window_size_list:
            pooled = layers.AveragePooling2D(pool_size=window_size, strides=1, padding='same')(inputs)
            deviation = pooled - inputs  # Deviation from average
            pooled_list.append(deviation)
        
        # Stack along new time dimension: (batch, time, height, width, channels)
        output = tf.stack(pooled_list, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.window_size_list), input_shape[1], input_shape[2], input_shape[3])
    
    def get_config(self):
        config = super().get_config()
        config['window_size_list'] = self.window_size_list
        return config


#################################################################################
# Model Architecture
#################################################################################

def create_vgg_feature_extractor():
    """
    Create VGG-style feature extractor (Featex) for ManTraNet.
    """
    img_in = layers.Input(shape=(None, None, 3))
    
    # Block 1: Combined conv (Bayar + SRM + regular) + regular conv
    x = CombinedConv2D(16, (5, 5), activation=None, name='b1c1')(img_in)
    x = layers.ReLU()(x)
    x = Conv2DSymPadding(32, 3, activation='relu', name='b1c2')(x)
    x = layers.AveragePooling2D(2, padding='same')(x)
    
    # Block 2
    x = Conv2DSymPadding(64, 3, activation='relu', name='b2c1')(x)
    x = Conv2DSymPadding(64, 3, activation='relu', name='b2c2')(x)
    x = layers.AveragePooling2D(2, padding='same')(x)
    
    # Block 3
    x = Conv2DSymPadding(128, 3, activation='relu', name='b3c1')(x)
    x = Conv2DSymPadding(128, 3, activation='relu', name='b3c2')(x)
    x = Conv2DSymPadding(128, 3, activation='relu', name='b3c3')(x)
    x = layers.AveragePooling2D(2, padding='same')(x)
    
    # Block 4
    x = Conv2DSymPadding(256, 3, activation='relu', name='b4c1')(x)
    x = Conv2DSymPadding(256, 3, activation='relu', name='b4c2')(x)
    x = Conv2DSymPadding(256, 3, activation='relu', name='b4c3')(x)
    x = layers.AveragePooling2D(2, padding='same')(x)
    
    # Block 5
    x = Conv2DSymPadding(256, 3, activation='relu', name='b5c1')(x)
    x = Conv2DSymPadding(256, 3, activation='relu', name='b5c2')(x)
    
    # Transform layer
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='transform')(x)
    
    return Model(inputs=img_in, outputs=x, name='Featex')


def create_mantranet_model():
    """
    Create complete ManTraNet model for image forgery detection.
    Returns model that takes RGB image and outputs manipulation mask.
    
    Note: Uses functional API with explicit shape to satisfy ConvLSTM2D requirements.
    """
    # Use a reasonable default size that will be resized during inference
    img_in = layers.Input(shape=(None, None, 3), name='img_in')
    
    # Feature extraction
    featex = create_vgg_feature_extractor()
    rf = featex(img_in)
    
    # Outlier transformation
    rf = layers.Conv2D(
        64, 1, 
        activation=None,
        use_bias=False,
        kernel_constraint=UnitNormConstraint(axis=-2),
        padding='same',
        name='outlierTrans'
    )(rf)
    
    # Batch normalization
    bf = layers.BatchNormalization(axis=-1, center=False, scale=False, name='bnorm')(rf)
    
    # Nested window averaging (creates temporal dimension)
    devf5d = NestedWindowAverageFeatExtractor(window_size_list=[7, 15, 31], name='nestedAvgFeatex')(bf)
    
    # Global standardization
    sigma = GlobalStd2D(name='glbStd')(bf)
    sigma5d = layers.Lambda(lambda t: K.expand_dims(t, axis=1), name='expTime')(sigma)
    devf5d = layers.Lambda(lambda vs: K.abs(vs[0] / (vs[1] + K.epsilon())), name='divStd')([devf5d, sigma5d])
    
    # ConvLSTM to aggregate temporal information
    devf = MantraNetConvLSTM(
        8, 7,
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        padding='same',
        return_sequences=False,
        name='cLSTM'
    )(devf5d)
    
    # Prediction head
    pred_out = layers.Conv2D(1, 7, padding='same', activation='sigmoid', name='pred')(devf)
    
    model = Model(inputs=img_in, outputs=pred_out, name='ManTraNet')
    return model


def load_mantranet_pretrained(model_dir, model_index=4):
    """
    Load pretrained ManTraNet model.
    
    Args:
        model_dir: Directory containing ManTraNet_Ptrain{index}.h5
        model_index: Model version (0-4), use 4 for best performance
        
    Returns:
        Loaded model ready for inference
    """
    weight_file = os.path.join(model_dir, f'ManTraNet_Ptrain{model_index}.h5')
    
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"Model weights not found: {weight_file}")
    
    # Build architecture
    model = create_mantranet_model()
    
    # Load weights
    model.load_weights(weight_file)
    
    return model
