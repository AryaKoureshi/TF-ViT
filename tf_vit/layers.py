import tensorflow as tf
from tensorflow import keras
from keras import layers

class PatchEmbedding3D(layers.Layer):
    def __init__(self, patch_size, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.conv = layers.Conv3D(self.hidden_dim, self.patch_size, strides=self.patch_size)
        self.flatten = layers.Reshape((-1, self.hidden_dim))

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        return x

class AddCLSToken(layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        return tf.concat([cls_tokens, inputs], axis=1)

class AddPositionEmbedding(layers.Layer):
    def __init__(self, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        _, seq_len, d_model = input_shape
        self.pos_embedding = self.add_weight(
            shape=(1, seq_len, d_model),
            initializer="random_normal",
            trainable=True,
            name="position_embedding"
        )
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        position_embeddings = self.pos_embedding
        return self.dropout(inputs + position_embeddings, training=training)

class TransformerEncoder(layers.Layer):
    def __init__(self, mlp_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=input_shape[-1] // self.num_heads
        )
        self.mlp = keras.Sequential([
            layers.Dense(self.mlp_dim, activation="gelu"),
            layers.Dense(input_shape[-1])
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        x = self.layernorm1(inputs)
        x = self.attention(x, x)
        x = self.dropout1(x, training=training)
        res = x + inputs

        x = self.layernorm2(res)
        x = self.mlp(x)
        x = self.dropout2(x, training=training)
        return x + res
