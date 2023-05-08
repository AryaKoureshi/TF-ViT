# coding=gbk

import os
from tqdm import tqdm
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from model import *
from data import *
from vit_keras import vit


#---------------------------------���ò���-------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CONFIG_B = {    # ViT-B����
    "hidden_dim":768,                              # patchǶ��ά��
    "liner_dim":3072,                              # mlp���Ա任ά��
    "atten_heads":12,                              # ע����ͷ��
    "encoder_depth":12,                            # �������ѵ�����
    }

CONFIG_L = {    # ViT-L����
    "hidden_dim":1024,     
    "liner_dim":4096,            
    "atten_heads":16,      
    "encoder_depth":24,    
    }

MODEL_CONFIG = CONFIG_B                            # ����ģ�͹�ģ
WEIGHT_CONFIG = "imagenet21k"                      # ����Ԥѵ��Ȩ��
                                                   #"imagenet21k"����pre_logits��Ȩ�أ��������21843������ߴ�Ϊ224
                                                   #"imagenet21k+imagenet2012" û��pre_logits��Ȩ�أ��������1000������ߴ�Ϊ384

IMAGE_SIZE = 224                                   # ͼƬ��С
PATCH_SIZE = 32                                    # patch��С
NUM_CLASSES = 10                                   # ������
DROPOUT_RATE = 0.1                                 # dropout����
ACTIVATION = "softmax"                             # ���ͷ�����
PRE_LOGITS = True                                  # �Ƿ����pre_logits��

LABEL_SMOOTH = 0.1                                 # ��ǩƽ��ϵ��
LEARNING_RATE = 1e-5                               # ��ʼѧϰ��
BATCH_SIZE = 32                                    # ѵ������֤������С
EPOCHS = 100                                       # ѵ������
STEPS_PER_EPOCH = 40000//BATCH_SIZE                # ÿ��ѵ����batch��
VALIDATION_STEPS = 10000//BATCH_SIZE               # ��֤��batch��
#-----------------------------------------------------------------------------


#---------------------------------����·��-------------------------------------
DATA_PATH = "datasets/cifar-10"
SAVE_PATH = f"save_models/vit-{'b' if MODEL_CONFIG==CONFIG_B else 'l'}_{PATCH_SIZE}_{WEIGHT_CONFIG}.h5"
PRE_TRAINED_PATH = f"pretrained/ViT-{'B' if MODEL_CONFIG==CONFIG_B else 'L'}_{PATCH_SIZE}_{WEIGHT_CONFIG}.npz"
#-----------------------------------------------------------------------------


#--------------------------------�������ݼ�-------------------------------------
train_data_gen, valid_data_gen = cifar_10_data_gen(
    path = DATA_PATH, 
    batch_size = BATCH_SIZE, 
    data = 'train', 
    resize = (IMAGE_SIZE,IMAGE_SIZE)
    )
#-----------------------------------------------------------------------------


#---------------------------------�ģ��-------------------------------------
vit = ViT(
    image_size = IMAGE_SIZE,
    patch_size = PATCH_SIZE,
    num_classes = NUM_CLASSES, 
    dropout_rate = DROPOUT_RATE,
    activation = ACTIVATION,
    pre_logits = PRE_LOGITS,
    **MODEL_CONFIG
    )

# ����Ԥѵ��Ȩ��
vit.load_pretrained_weights(PRE_TRAINED_PATH)

vit.compile(
    optimizer=SGD(LEARNING_RATE, momentum=0.9),
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['acc']
    )
vit.summary()
#-----------------------------------------------------------------------------


#--------------------------------ѵ���ͱ���-------------------------------------
best_loss, _ = vit.evaluate(valid_data_gen, steps=VALIDATION_STEPS)
stop = 0
for e in range(EPOCHS):
    e_loss, e_acc = 0, 0
    for b in tqdm(range(STEPS_PER_EPOCH), f"training"):
        (x_batch, y_batch) = next(train_data_gen)
        loss, acc = vit.train_on_batch(x_batch, y_batch)
        e_loss += loss
        e_acc += acc
    print(f"[epoch:{e}]\
        \t[loss:{round(e_loss/STEPS_PER_EPOCH,4)}]\
        \t[acc:{round(e_acc/STEPS_PER_EPOCH,4)}]\
        \t[lr:{K.get_value(vit.optimizer.lr)}]") 
    loss, acc = vit.evaluate(valid_data_gen, steps=VALIDATION_STEPS)
    if loss < best_loss:
        best_loss = loss
        stop = 0
        vit.save_weights(SAVE_PATH)
        print(f"model saved")
    else:
        stop += 1
        if stop >= 3:
            print(f"early stop with eopch {e}")
            break
    lr = K.get_value(vit.optimizer.lr)
    K.set_value(vit.optimizer.lr, lr*0.9)
    print("-"*100)
#-----------------------------------------------------------------------------