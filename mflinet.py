from tensorflow.keras.layers import Input, Dense, LayerNormalization, Reshape, Conv2D, MultiHeadAttention, Layer
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow_addons as tfa

# Differential Attention Layer with learnable lambda
class DifferentialAttention(Layer):
    def __init__(self, num_heads, key_dim, lambda_init=0.8, **kwargs):
        super(DifferentialAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.lambda_init = lambda_init
        
    def build(self, input_shape):
        # Create learnable lambda parameter
        self.lambda_param = self.add_weight(
            name='lambda',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.lambda_init),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg() 
        )
        
        # Create two separate attention mechanisms
        self.attention_1 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.attention_2 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        
        super(DifferentialAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Calculate separate attention maps
        attn_output1 = self.attention_1(query=inputs, value=inputs, key=inputs)
        attn_output2 = self.attention_2(query=inputs, value=inputs, key=inputs)
        
        # Differential attention with learnable lambda
        differential_output = attn_output1 - self.lambda_param * attn_output2
        return differential_output
    
    def get_config(self):
        config = super(DifferentialAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'lambda_init': self.lambda_init
        })
        return config

# Define DIFF Transformer Encoder Block
def diff_transformer_encoder_block(input_tensor, num_heads, key_dim, ff_dim, lambda_init=0.8):
    # Differential attention with learnable lambda
    attn_output = DifferentialAttention(num_heads, key_dim, lambda_init)(input_tensor)
    norm_attn_output = tfa.layers.GroupNormalization(groups=4)(input_tensor + attn_output)

    # Feed-forward network with SwiGLU
    ff_output = Dense(ff_dim * 2, activation="swish")(norm_attn_output)
    ff_output = Dense(ff_dim)(ff_output)
    ff_output = LayerNormalization()(norm_attn_output + ff_output)
    return ff_output

# Define DIFF Transformer Decoder Block with Cross-Attention
def diff_transformer_decoder_block(input_tensor, encoder_output, num_heads, key_dim, ff_dim, lambda_init=0.8):
    # Self-attention with learnable lambda
    self_attn_output = DifferentialAttention(num_heads, key_dim, lambda_init)(input_tensor)
    norm_self_attn_output = tfa.layers.GroupNormalization(groups=4)(input_tensor + self_attn_output)

    # Cross-attention with encoder output
    cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    cross_attn_output = cross_attention(query=norm_self_attn_output, value=encoder_output, key=encoder_output)
    norm_cross_attn_output = tfa.layers.GroupNormalization(groups=4)(norm_self_attn_output + cross_attn_output)

    # Feed-forward network with SwiGLU
    ff_output = Dense(ff_dim * 2, activation="swish")(norm_cross_attn_output)
    ff_output = Dense(ff_dim)(ff_output)
    ff_output = tfa.layers.GroupNormalization(groups=4)(norm_cross_attn_output + ff_output)
    return ff_output

# Define the input layers
input_1 = Input(shape=(1, 1, 176))
input_2 = Input(shape=(1, 1, 176))

# Encoder blocks for each input
encoder_output1 = diff_transformer_encoder_block(input_1, num_heads=16, key_dim=176, ff_dim=176)
encoder_output1 = diff_transformer_encoder_block(encoder_output1, num_heads=16, key_dim=176, ff_dim=176)

encoder_output2 = diff_transformer_encoder_block(input_2, num_heads=16, key_dim=176, ff_dim=176)
encoder_output2 = diff_transformer_encoder_block(encoder_output2, num_heads=16, key_dim=176, ff_dim=176)

# Branch 1: Decoder for first output with cross-attention
decoder_output1 = diff_transformer_decoder_block(encoder_output1, encoder_output2, num_heads=16, key_dim=176, ff_dim=176)
decoder_output1_reshaped = Reshape((1, 1, 176))(decoder_output1)
output1 = Conv2D(1, kernel_size=(1, 1), activation="elu", padding="same", kernel_regularizer="l2")(decoder_output1_reshaped)

# Branch 2: Decoder for second output with cross-attention
decoder_output2 = diff_transformer_decoder_block(encoder_output1, encoder_output2, num_heads=16, key_dim=176, ff_dim=176)
decoder_output2_reshaped = Reshape((1, 1, 176))(decoder_output2)
output2 = Conv2D(1, kernel_size=(1, 1), activation="elu", padding="same", kernel_regularizer="l2")(decoder_output2_reshaped)

# Branch 3: Another cross-attended branch for the third output
decoder_output3 = diff_transformer_decoder_block(encoder_output1, encoder_output2, num_heads=16, key_dim=176, ff_dim=176)
decoder_output3_reshaped = Reshape((1, 1, 176))(decoder_output3)
output3 = Conv2D(1, kernel_size=(1, 1), activation="elu", padding="same", kernel_regularizer="l2")(decoder_output3_reshaped)

# Define the final model with two inputs and three outputs
modelD = Model(inputs=[input_1, input_2], outputs=[output1, output2, output3])
modelD.compile(optimizer="rmsprop", loss="mse", metrics=["mse"])

# Display the model summary
modelD.summary()
