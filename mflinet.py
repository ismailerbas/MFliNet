"""
MFliNet: Macroscopic Fluorescence Lifetime Imaging Network
===========================================================

A Differential Transformer encoder-decoder architecture for Fluorescence Lifetime Imaging (FLI)
parameter estimation that processes both TPSF (Time-Point Spread Function) and pixel-wise IRF 
(Instrument Response Function) inputs to predict fluorescence lifetime components.

Architecture outputs:
- τ₁: Short lifetime component (0.2-0.8 ns)
- τ₂: Long lifetime component (0.8-1.5 ns)  
- Fractional amplitude

References:
    Erbas et al. (2024). "Enhancing fluorescence lifetime parameter estimation accuracy with 
    differential transformer based deep learning model incorporating pixelwise instrument 
    response function." arXiv preprint arXiv:2411.16896.

Copyright 2025 RPI. Licensed under Apache License 2.0.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Reshape, Conv2D, MultiHeadAttention, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa


class LearnableLambda(Layer):
    """
    Learnable lambda parameter for adaptive noise cancellation in differential attention.
    
    This trainable parameter controls the suppression strength of the second attention map,
    enabling the model to adaptively balance signal amplification and noise reduction based
    on training data characteristics.
    
    Attributes:
        lambda_param: Trainable scalar parameter initialized to 0.8 (typical range: 0.5-1.0)
    """
    
    def __init__(self, initial_value=0.8, **kwargs):
        """
        Initialize the learnable lambda parameter.
        
        Args:
            initial_value: Initial value for lambda parameter (default: 0.8 based on paper)
        """
        super(LearnableLambda, self).__init__(**kwargs)
        self.initial_value = initial_value
    
    def build(self, input_shape):
        """
        Create the trainable lambda weight.
        
        Args:
            input_shape: Shape of input tensor (unused, required by Keras API)
        """
        # Initialize lambda as a trainable weight with constraint to ensure stability
        self.lambda_param = self.add_weight(
            name='lambda_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_value),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=1.5)
        )
        super(LearnableLambda, self).build(input_shape)
    
    def call(self, inputs):
        """
        Return the current lambda value.
        
        Args:
            inputs: Input tensor (unused, maintained for Keras compatibility)
            
        Returns:
            Current value of lambda parameter
        """
        return self.lambda_param
    
    def get_config(self):
        """Return configuration for model serialization."""
        config = super(LearnableLambda, self).get_config()
        config.update({'initial_value': self.initial_value})
        return config


def differential_attention(input_tensor, num_heads, key_dim, learnable_lambda_layer):
    """
    Differential attention mechanism that amplifies signal while suppressing noise.
    
    This mechanism computes two separate multi-head attention maps and subtracts the second
    from the first, weighted by a learnable lambda parameter. This subtractive operation
    helps cancel common noise patterns while preserving signal-specific features.
    
    Mathematical formulation:
        Attention_diff = Attention_1 - λ * Attention_2
    
    where λ is learned during training to optimize noise suppression.
    
    Args:
        input_tensor: Input tensor of shape (batch_size, sequence_length, embedding_dim)
        num_heads: Number of attention heads for parallel attention computations
        key_dim: Dimensionality of key/query projections per attention head
        learnable_lambda_layer: Instance of LearnableLambda for adaptive noise cancellation
        
    Returns:
        Differential attention output with same shape as input_tensor
    """
    # First attention path: Primary signal extraction
    # Self-attention captures dependencies between all positions in the sequence
    attention_1 = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim,
        name='attention_primary'
    )
    
    # Second attention path: Noise pattern extraction  
    # Independent attention heads with separate learnable weights
    attention_2 = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim,
        name='attention_noise'
    )
    
    # Compute primary attention map using self-attention
    # Query, Key, and Value are all derived from the same input tensor
    attn_output1 = attention_1(
        query=input_tensor, 
        value=input_tensor, 
        key=input_tensor
    )
    
    # Compute secondary attention map for noise characterization
    attn_output2 = attention_2(
        query=input_tensor, 
        value=input_tensor, 
        key=input_tensor
    )
    
    # Get current learnable lambda value
    lambda_value = learnable_lambda_layer(input_tensor)
    
    # Apply differential attention: Subtract weighted noise map from signal map
    # This enhances signal-to-noise ratio by canceling common patterns
    differential_output = attn_output1 - lambda_value * attn_output2
    
    return differential_output


def diff_transformer_encoder_block(input_tensor, num_heads, key_dim, ff_dim, learnable_lambda_layer):
    """
    DIFF Transformer encoder block with differential attention and feed-forward network.
    
    Each encoder block consists of:
    1. Differential attention sublayer with residual connection
    2. Group normalization for stabilizing activations
    3. Position-wise feed-forward network with SwiGLU activation
    4. Layer normalization with residual connection
    
    This architecture processes TPSF or IRF input to extract hierarchical temporal features
    relevant for fluorescence lifetime estimation.
    
    Args:
        input_tensor: Input tensor of shape (batch_size, 1, 1, 176) representing time gates
        num_heads: Number of attention heads (16 for 176-dimensional inputs)
        key_dim: Dimension of attention keys/queries (176 to match input dimension)
        ff_dim: Hidden dimension of feed-forward network (176 maintains dimension)
        learnable_lambda_layer: Shared learnable lambda parameter for noise suppression
        
    Returns:
        Encoded representation with same shape as input_tensor
    """
    # Apply differential attention mechanism
    # Captures temporal dependencies in fluorescence decay curves
    attn_output = differential_attention(
        input_tensor, 
        num_heads, 
        key_dim, 
        learnable_lambda_layer
    )
    
    # Residual connection: Add attention output to original input
    # Helps gradient flow and preserves input information
    residual_attn = input_tensor + attn_output
    
    # Group normalization: Normalize over groups of channels
    # More stable than batch normalization for small batch sizes
    # groups=4 divides 176 channels into groups of 44
    norm_attn_output = tfa.layers.GroupNormalization(
        groups=4, 
        name='encoder_group_norm_attn'
    )(residual_attn)
    
    # Feed-forward network (FFN) with SwiGLU activation
    # First layer: Expand to 2x dimension for gating mechanism
    # Swish activation (x * sigmoid(x)) provides smooth non-linearity
    ff_output = Dense(
        ff_dim * 2, 
        activation='swish',
        name='encoder_ffn_expand'
    )(norm_attn_output)
    
    # Second layer: Project back to original dimension
    # Compresses expanded representation while maintaining expressiveness
    ff_output = Dense(
        ff_dim,
        name='encoder_ffn_project'
    )(ff_output)
    
    # Residual connection and layer normalization for FFN sublayer
    # Stabilizes training and improves gradient propagation
    ff_output = LayerNormalization(
        name='encoder_layer_norm_ffn'
    )(norm_attn_output + ff_output)
    
    return ff_output


def diff_transformer_decoder_block(input_tensor, encoder_output, num_heads, key_dim, ff_dim, learnable_lambda_layer):
    """
    DIFF Transformer decoder block with self-attention, cross-attention, and feed-forward network.
    
    Each decoder block implements:
    1. Differential self-attention on decoder input with residual connection
    2. Group normalization
    3. Cross-attention between decoder and encoder representations
    4. Group normalization  
    5. Feed-forward network with SwiGLU activation
    6. Group normalization with residual connection
    
    Cross-attention enables the decoder to focus on relevant encoder features (IRF characteristics)
    while predicting lifetime parameters from TPSF data.
    
    Args:
        input_tensor: Decoder input tensor (encoded TPSF features)
        encoder_output: Encoder output tensor (encoded IRF features)  
        num_heads: Number of attention heads
        key_dim: Dimension of attention keys/queries
        ff_dim: Hidden dimension of feed-forward network
        learnable_lambda_layer: Shared learnable lambda for differential attention
        
    Returns:
        Decoded representation ready for parameter prediction heads
    """
    # Self-attention sublayer: Decoder attends to its own previous representations
    # Captures dependencies within TPSF-derived features
    self_attn_output = differential_attention(
        input_tensor, 
        num_heads, 
        key_dim, 
        learnable_lambda_layer
    )
    
    # Residual connection preserves gradient flow through decoder layers
    residual_self_attn = input_tensor + self_attn_output
    
    # Normalize self-attention output for stable cross-attention computation
    norm_self_attn_output = tfa.layers.GroupNormalization(
        groups=4,
        name='decoder_group_norm_self_attn'
    )(residual_self_attn)
    
    # Cross-attention sublayer: Decoder queries encoder representations
    # Query comes from decoder (TPSF features), Key and Value from encoder (IRF features)
    # This allows pixel-wise IRF information to modulate TPSF interpretation
    cross_attention = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim,
        name='decoder_cross_attention'
    )
    
    cross_attn_output = cross_attention(
        query=norm_self_attn_output,  # What the decoder is looking for
        value=encoder_output,          # Information from encoder (IRF)
        key=encoder_output             # Addressing mechanism for encoder features
    )
    
    # Residual connection for cross-attention
    residual_cross_attn = norm_self_attn_output + cross_attn_output
    
    # Normalize before feed-forward network
    norm_cross_attn_output = tfa.layers.GroupNormalization(
        groups=4,
        name='decoder_group_norm_cross_attn'
    )(residual_cross_attn)
    
    # Position-wise feed-forward network with SwiGLU
    # Expands to 2x dimension for gating, then projects back
    ff_output = Dense(
        ff_dim * 2, 
        activation='swish',
        name='decoder_ffn_expand'
    )(norm_cross_attn_output)
    
    ff_output = Dense(
        ff_dim,
        name='decoder_ffn_project'
    )(ff_output)
    
    # Final residual connection and normalization
    ff_output = tfa.layers.GroupNormalization(
        groups=4,
        name='decoder_group_norm_ffn'
    )(norm_cross_attn_output + ff_output)
    
    return ff_output


# ============================================================================
# MODEL ARCHITECTURE DEFINITION
# ============================================================================

# Input layers for dual-input architecture
# Input 1: TPSF (Time-Point Spread Function) - experimental photon time-of-arrival histogram
# Shape: (batch_size, 1, 1, 176) where 176 represents time gates
input_1 = Input(shape=(1, 1, 176), name='tpsf_input')

# Input 2: Pixel-wise IRF (Instrument Response Function)
# Shape: (batch_size, 1, 1, 176) matching TPSF temporal resolution
input_2 = Input(shape=(1, 1, 176), name='irf_input')

# Initialize learnable lambda parameter (shared across all attention layers)
# Enables adaptive noise cancellation strength based on training data
learnable_lambda = LearnableLambda(initial_value=0.8, name='learnable_lambda')

# ============================================================================
# ENCODER STREAM 1: TPSF Processing
# ============================================================================
# First encoder block: Extract low-level temporal features from TPSF
encoder_output1 = diff_transformer_encoder_block(
    input_1, 
    num_heads=16,      # 16 heads for 176-dim input (11 dims per head)
    key_dim=176,       # Match input dimension for full representation
    ff_dim=176,        # Maintain dimensionality through network
    learnable_lambda_layer=learnable_lambda
)

# Second encoder block: Extract high-level temporal features from TPSF
# Stacking enables hierarchical feature learning
encoder_output1 = diff_transformer_encoder_block(
    encoder_output1, 
    num_heads=16, 
    key_dim=176, 
    ff_dim=176,
    learnable_lambda_layer=learnable_lambda
)

# ============================================================================
# ENCODER STREAM 2: IRF Processing
# ============================================================================
# First encoder block: Extract low-level IRF characteristics
# IRF captures system response and pixel-wise variations
encoder_output2 = diff_transformer_encoder_block(
    input_2, 
    num_heads=16, 
    key_dim=176, 
    ff_dim=176,
    learnable_lambda_layer=learnable_lambda
)

# Second encoder block: Extract high-level IRF features
encoder_output2 = diff_transformer_encoder_block(
    encoder_output2, 
    num_heads=16, 
    key_dim=176, 
    ff_dim=176,
    learnable_lambda_layer=learnable_lambda
)

# ============================================================================
# DECODER BRANCH 1: τ₁ Prediction (Short Lifetime Component)
# ============================================================================
# Cross-attend TPSF features with IRF features for τ₁ estimation
# Range: 0.2-0.8 ns (fast decay component)
decoder_output1 = diff_transformer_decoder_block(
    encoder_output1,           # Query: TPSF-derived features
    encoder_output2,           # Key/Value: IRF-derived features
    num_heads=16, 
    key_dim=176, 
    ff_dim=176,
    learnable_lambda_layer=learnable_lambda
)

# Reshape for convolutional prediction head
decoder_output1_reshaped = Reshape(
    (1, 1, 176),
    name='reshape_tau1'
)(decoder_output1)

# Prediction head: 1x1 convolution maps features to τ₁ estimate
# ELU activation ensures positive lifetime values
# L2 regularization prevents overfitting
output1 = Conv2D(
    filters=1,
    kernel_size=(1, 1), 
    activation='elu',
    padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(0.01),
    name='output_tau1'
)(decoder_output1_reshaped)

# ============================================================================
# DECODER BRANCH 2: τ₂ Prediction (Long Lifetime Component)
# ============================================================================
# Independent decoder for τ₂ estimation with same architecture
# Range: 0.8-1.5 ns (slow decay component)
decoder_output2 = diff_transformer_decoder_block(
    encoder_output1, 
    encoder_output2, 
    num_heads=16, 
    key_dim=176, 
    ff_dim=176,
    learnable_lambda_layer=learnable_lambda
)

decoder_output2_reshaped = Reshape(
    (1, 1, 176),
    name='reshape_tau2'
)(decoder_output2)

output2 = Conv2D(
    filters=1,
    kernel_size=(1, 1), 
    activation='elu',
    padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(0.01),
    name='output_tau2'
)(decoder_output2_reshaped)

# ============================================================================
# DECODER BRANCH 3: Fractional Amplitude Prediction
# ============================================================================
# Third decoder branch for estimating relative contribution of τ₁ component
# Output range: [0, 1] representing fraction of short lifetime component
decoder_output3 = diff_transformer_decoder_block(
    encoder_output1, 
    encoder_output2, 
    num_heads=16, 
    key_dim=176, 
    ff_dim=176,
    learnable_lambda_layer=learnable_lambda
)

decoder_output3_reshaped = Reshape(
    (1, 1, 176),
    name='reshape_amplitude'
)(decoder_output3)

output3 = Conv2D(
    filters=1,
    kernel_size=(1, 1), 
    activation='elu',
    padding='same',
    kernel_regularizer=tf.keras.regularizers.l2(0.01),
    name='output_amplitude'
)(decoder_output3_reshaped)

# ============================================================================
# MODEL COMPILATION
# ============================================================================
# Define complete MFliNet model with dual inputs and three outputs
modelD = Model(
    inputs=[input_1, input_2], 
    outputs=[output1, output2, output3],
    name='MFliNet'
)

# Configure Adam optimizer as specified
# Learning rate: 0.001 (adaptive learning rate during training)
# No weight decay (removed AdamW functionality)
optimizer = Adam(learning_rate=0.001)

# Compile model with MSE loss for each output branch
# MSE appropriate for continuous lifetime parameter regression
# Metrics track training progress for each output separately
modelD.compile(
    optimizer=optimizer,
    loss=['mse', 'mse', 'mse'],  # Separate loss for τ₁, τ₂, and amplitude
    metrics=['mse']               # Track MSE for each output
)

# Display model architecture summary
# Shows layer connectivity, parameter counts, and output shapes
modelD.summary()

# ============================================================================
# MODEL TRAINING EXAMPLE
# ============================================================================
"""
# Training requires data from SS_RLD pipeline
# See: https://github.com/vkp217/SS_RLD/tree/main/src/ss

# Assuming train_data is a tuple: ([tpsf_train, irf_train], [tau1_train, tau2_train, amp_train])
# And val_data is structured similarly

history = modelD.fit(
    x=[tpsf_train, irf_train],
    y=[tau1_train, tau2_train, amp_train],
    validation_data=([tpsf_val, irf_val], [tau1_val, tau2_val, amp_val]),
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('results/models/mflinet_best.h5', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    ]
)

# Save trained model
modelD.save('results/models/mflinet_final.h5')

# Inference on experimental data
predictions = modelD.predict([tpsf_experimental, irf_experimental])
tau1_pred, tau2_pred, amplitude_pred = predictions
"""
