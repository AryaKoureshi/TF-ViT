# coding=gbk

import os
from model import *
from data import *


#---------------------------------���ò���-------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CONFIG_B = {
    "hidden_dim":768,                              # patchǶ��ά��
    "liner_dim":3072,                              # mlp���Ա任ά��
    "atten_heads":12,                              # ע����ͷ��
    "encoder_depth":12,                            # �������ѵ�����
    }

CONFIG_L = {    
    "hidden_dim":1024,     
    "liner_dim":4096,            
    "atten_heads":16,      
    "encoder_depth":24,    
    }

MODEL_CONFIG = CONFIG_B                            # ����ģ�͹�ģ
WEIGHT_CONFIG = "imagenet21k"                      # ����Ԥѵ��Ȩ��

IMAGE_SIZE = 224                                   # ͼƬ��С
PATCH_SIZE = 32                                    # patch��С
NUM_CLASSES = 10                                   # ������
DROPOUT_RATE = 0.1,                                # dropout����
ACTIVATION = "softmax"                             # ���ͷ�����
PRE_LOGITS = True                                  # �Ƿ����pre_logits��

BATCH_SIZE = 10                                    # ���Ե�����С
STEPS=10000//BATCH_SIZE                            # ���Լ���batch��
#-----------------------------------------------------------------------------


#---------------------------------����·��-------------------------------------
SAVE_PATH = f"save_models/vit-{'b' if MODEL_CONFIG==CONFIG_B else 'l'}_{PATCH_SIZE}_{WEIGHT_CONFIG}.h5"
DATA_PATH = "datasets/cifar-10"
#-----------------------------------------------------------------------------


#--------------------------------�������ݼ�-------------------------------------
test_data_gen = cifar_10_data_gen(
    path = DATA_PATH, 
    batch_size = BATCH_SIZE, 
    data = 'test', 
    resize = (IMAGE_SIZE,IMAGE_SIZE)
    )
#-----------------------------------------------------------------------------


#---------------------------------����ģ��-------------------------------------
vit = ViT(
    image_size = IMAGE_SIZE,
    patch_size = PATCH_SIZE,
    num_classes = NUM_CLASSES, 
    dropout_rate = DROPOUT_RATE,
    activation = ACTIVATION,
    pre_logits = PRE_LOGITS,
    **MODEL_CONFIG)
vit.load_weights(SAVE_PATH)
vit.summary()
#-----------------------------------------------------------------------------


#----------------------------------����ģʽ-------------------------------------
vit.compile(
    loss='categorical_crossentropy', 
    metrics=['acc']
    )
vit.evaluate(
    test_data_gen, 
    steps=STEPS
    )
#-----------------------------------------------------------------------------