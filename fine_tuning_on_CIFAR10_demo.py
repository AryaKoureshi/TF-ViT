from tqdm import tqdm
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from keras import backend as K
from data import cifar_10_data_gen
from keras_vit.vit import ViT_B32, ViT


#-----------------------------------------------------------------------------
WEIGHT_CONFIG = "imagenet21k"            
NUM_CLASSES = 10                
DROPOUT_RATE = 0.1                         
ACTIVATION = "softmax"             

LABEL_SMOOTH = 0.1                        
LEARNING_RATE = 1e-3                  
BATCH_SIZE = 16                               
EPOCHS = 100                                    
STEPS_PER_EPOCH = 40000//BATCH_SIZE               
VALIDATION_STEPS = 10000//BATCH_SIZE           
TEST_STEPS = 10000//BATCH_SIZE                     
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
DATA_PATH = "datasets/cifar-10"
SAVE_PATH = f"save_models/"
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
vit = ViT_B32(
    num_classes = NUM_CLASSES, 
    activation = ACTIVATION,
    dropout_rate = DROPOUT_RATE,
    weights=WEIGHT_CONFIG,
    )

vit.summary()
vit.loading_summary()

vit.compile(
    optimizer=SGD(LEARNING_RATE, momentum=0.9),
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['acc']
    )
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
train_data_gen, valid_data_gen = cifar_10_data_gen(
    path = DATA_PATH, 
    batch_size = BATCH_SIZE, 
    data = 'train', 
    resize = vit.image_size
    )
test_data_gen = cifar_10_data_gen(
    path = DATA_PATH, 
    batch_size = BATCH_SIZE, 
    data = 'test', 
    resize = vit.image_size
    )
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
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
        vit.save_weights(SAVE_PATH+vit.name+".h5")
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


#-----------------------------------------------------------------------------
vit.evaluate(
    test_data_gen, 
    steps=TEST_STEPS
    )
#-----------------------------------------------------------------------------
