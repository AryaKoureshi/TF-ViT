# coding=gbk

import tensorflow as tf
import keras.backend as K
from keras import layers
from keras import models
from keras import activations


class PatchEmbedding(layers.Layer):
    '''PatchǶ���'''
    def __init__(self, patch_size, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.conv = layers.Conv2D(self.hidden_dim, self.patch_size, self.patch_size)
        self.flatten = layers.Reshape((-1, self.hidden_dim))
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        return x


class AddCLSToken(layers.Layer):
    '''classtoken��Ӳ�'''
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.class_token = self.add_weight(
            name="class_token_weight",
            shape=(1, 1, self.hidden_dim),
            initializer="zero",
            trainable=True
            )
        return super().build(input_shape)

    def call(self, inputs):   
        x = tf.tile(self.class_token, [K.shape(inputs)[0],1,1])
        x = tf.concat([x, inputs], axis=1)
        return x


class AddPositionEmbedding(layers.Layer):
    '''λ��Ƕ���'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.pe = self.add_weight(
            name="position_embedding",
            shape=(1, input_shape[1], input_shape[2]),
            initializer='random_normal',
            trainable=True
            )
        return super().build(input_shape)

    def call(self, inputs):
        return inputs + self.pe


class MultiHeadAttention(layers.Layer):
    '''��ͷע������'''
    def __init__(
        self, 
        heads,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.heads = heads

    def build(self, input_shape):
        self.dk = tf.sqrt(tf.cast(input_shape[-1]//self.heads, dtype=tf.float32))
        self.q_dense = layers.Dense(input_shape[-1], name="query")
        self.k_dense = layers.Dense(input_shape[-1], name="key")
        self.v_dense = layers.Dense(input_shape[-1], name="value")
        self.o_dense = layers.Dense(input_shape[-1], name="combine_out")
        return super().build(input_shape)

    def call(self, inputs):
        q = self.q_dense(inputs)
        k = self.k_dense(inputs)
        v = self.v_dense(inputs)
        q = tf.concat(tf.split(q, self.heads, axis=-1), axis=0)
        k = tf.concat(tf.split(k, self.heads, axis=-1), axis=0)
        v = tf.concat(tf.split(v, self.heads, axis=-1), axis=0)
        qk = tf.matmul(q, k, transpose_b=True)  
        qk = K.softmax(qk / self.dk)
        qkv = tf.matmul(qk, v)
        qkv = tf.concat(tf.split(qkv, self.heads, axis=0), axis=-1)
        return self.o_dense(qkv)


class MLPBlock(layers.Layer):
    '''mlp block'''
    def __init__(self, liner_dim, dropout_rate, **kwargs):
        self.liner_dim = liner_dim
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.dropout = layers.Dropout(self.dropout_rate)
        self.liner_1 = layers.Dense(self.liner_dim, activations.gelu)
        self.liner_2 = layers.Dense(input_shape[-1])
        return super().build(input_shape)

    def call(self, inputs):
        h = self.liner_1(inputs)
        h = self.dropout(h)
        h = self.liner_2(h)
        h = self.dropout(h)
        return h


class TransformerEncoder(layers.Layer):
    '''Transformerd Encoder��'''
    def __init__(
        self, 
        liner_dim,
        atten_heads,
        dropout_rate,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.liner_dim = liner_dim
        self.atten_heads = atten_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):    
        self.multi_head_attens = MultiHeadAttention(
            name='multi_head_attention_layer',
            heads=self.atten_heads, 
            )
        self.mlp_block = MLPBlock(
            name='mlp_block_layer',
            liner_dim=self.liner_dim,
            dropout_rate = self.dropout_rate
            )
        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs):
        x = self.layer_norm_1(inputs)
        x = self.multi_head_attens(x)
        x = self.dropout(x)
        x = x + inputs
        y = self.layer_norm_2(x)
        y = self.mlp_block(y)
        return x + y


class ViT(models.Model):
    '''VisionTransformerʵ��
    ������
        - image_size������ͼ���С
        - patch_size��ÿ��patch�ĳߴ�
        - num_classes�������������
        - hidden_dim��ÿ��patch��embeddingά�ȣ�Ϊatten_heads������
        - liner_dim��mlpģ���ͶӰά�ȣ�һ����h_dim��4��
        - atten_heads��ע����ͷ������
        - encoder_depth������������
        - dropout_rate��dropout���ʣ�Ĭ��Ϊ0.1
        - activatiion�����ͷ�ļ����
        - pre_logits���Ƿ����pre_logits��
        - include_mlp_head���Ƿ����mlp_head��
    '''
    def __init__(
        self, 
        image_size,
        patch_size,
        num_classes,
        hidden_dim,
        liner_dim,
        atten_heads,
        encoder_depth,
        dropout_rate=0.1,
        activation="linear",
        pre_logits=True,
        include_mlp_head=True,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.hidden_dim = hidden_dim
        self.liner_dim = liner_dim
        self.num_classes = num_classes
        self.atten_heads = atten_heads
        self.encoder_depth = encoder_depth
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.pre_logits = pre_logits
        self.include_mlp_head = include_mlp_head
        assert hidden_dim%atten_heads==0, "hidden_dim��atten_heads��ƥ�䣡"
        self.build((None, *self.image_size, 3))

    def build(self, input_shape):
        self.patch_embedding=PatchEmbedding(self.patch_size, self.hidden_dim, name="patch_embedding")
        self.add_cls_token = AddCLSToken(self.hidden_dim, name="add_cls_token")
        self.position_embedding = AddPositionEmbedding(name="add_position_embedding")
        self.encoder_blocks = [
            TransformerEncoder(
                self.liner_dim, 
                self.atten_heads, 
                self.dropout_rate,
                name=f"transformer_block_{i}"
                ) for i in range(self.encoder_depth)]
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
        self.extract_token = layers.Lambda(lambda x: x[:,0], name="extract_token")
        if self.pre_logits: self.pre_logits = layers.Dense(self.hidden_dim, activation="tanh", name="pre_logits")
        if self.include_mlp_head: self.mlp_head = layers.Dense(self.num_classes, self.activation, name="mlp_head")
        super().build(input_shape)
        self.call(layers.Input(input_shape[1:]))

    def call(self, inputs):
        x = self.patch_embedding(inputs)
        x = self.add_cls_token(x)
        x = self.position_embedding(x)
        for encoder in self.encoder_blocks:
            x = encoder(x)
        x = self.layer_norm(x)
        x = self.extract_token(x)
        if self.pre_logits:
            x = self.pre_logits(x)
        if self.include_mlp_head:
            x = self.mlp_head(x)
        return x

    def load_pretrained_weights(self, weights_npz):
        params_dict = K.np.load(weights_npz, allow_pickle=False)
        keys = list(params_dict.keys())
        transformer_blocks_num = len([k for k in keys if "Transformer/encoderblock" in k])//16
        # print("��Ȩ���ļ������Ĳ㣺\n", keys)

        # patch_embedding��Ȩ��
        patch_emb_weights = [
            params_dict["embedding/kernel"],
            params_dict["embedding/bias"]]

        # cls_token��Ȩ��
        cls_token_weights = [params_dict["cls"]]

        # position_embedding��Ȩ��
        position_emb_weights = [params_dict["Transformer/posembed_input/pos_embedding"]]

        # transformer_block��Ȩ��
        transformer_block_weights = [
            [params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel"], 
             params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias"],
             params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel"], 
             params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias"], 
             params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel"], 
             params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias"], 
             params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel"], 
             params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias"], 
             params_dict[f"Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel"], 
             params_dict[f"Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias"], 
             params_dict[f"Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel"], 
             params_dict[f"Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias"], 
             params_dict[f"Transformer/encoderblock_{i}/LayerNorm_0/scale"], 
             params_dict[f"Transformer/encoderblock_{i}/LayerNorm_0/bias"], 
             params_dict[f"Transformer/encoderblock_{i}/LayerNorm_2/scale"], 
             params_dict[f"Transformer/encoderblock_{i}/LayerNorm_2/bias"]
             ]  for i in range(transformer_blocks_num)]

        # ���layer_norm��Ȩ��
        layer_norm_weights = [
            params_dict["Transformer/encoder_norm/scale"],
            params_dict["Transformer/encoder_norm/bias"]]

        # ���pre_logic��Ȩ��
        if "pre_logits/kernel" in keys:
            pre_logic_weights = [
                params_dict["pre_logits/kernel"],
                params_dict["pre_logits/bias"]]

        # ���head��Ȩ��
        mlp_head_weights = [
            params_dict["head/kernel"],
            params_dict["head/bias"]]

        match_layer, dismatch_layer = [], []
        for l in self.layers:
            
            # ����patch_embedding��Ȩ��
            if isinstance(l, PatchEmbedding):
                try:
                    l.set_weights(patch_emb_weights)
                    match_layer.append(l.name)
                except: dismatch_layer.append(l.name)

            # ����cls_token��Ȩ��
            elif isinstance(l, AddCLSToken):
                try:
                    l.set_weights(cls_token_weights)
                    match_layer.append(l.name)
                except: dismatch_layer.append(l.name)

            # ����position_embedding��Ȩ��
            elif isinstance(l, AddPositionEmbedding):
                try:
                    l.set_weights(position_emb_weights)
                    match_layer.append(l.name)
                except: dismatch_layer.append(l.name)

            # ����transformer_block��Ȩ��
            elif isinstance(l, TransformerEncoder):
                i = int(l.name[l.name.rfind("_")+1:])
                try:
                    weights = [w.reshape(s.shape) for w,s in zip(transformer_block_weights[i], l.weights)]
                    l.set_weights(weights)
                    match_layer.append(l.name)
                except: dismatch_layer.append(l.name)

            # ����layer_norm��Ȩ��
            elif isinstance(l, layers.LayerNormalization):
                try:
                    l.set_weights(layer_norm_weights)
                    match_layer.append(l.name)
                except: dismatch_layer.append(l.name)

            # ����pre_logits��Ȩ��
            elif l.name == "pre_logits":
                try:
                    l.set_weights(pre_logic_weights)
                    match_layer.append(l.name)
                except: dismatch_layer.append(l.name)

            # ����mlp_head��Ȩ��
            elif l.name == "mlp_head":
                try:
                    l.set_weights(mlp_head_weights)
                    match_layer.append(l.name)
                except: dismatch_layer.append(l.name)

        print("����Ȩ�سɹ���>>", match_layer)
        print("����Ȩ��ʧ�ܲ�>>", dismatch_layer)
        return match_layer, dismatch_layer