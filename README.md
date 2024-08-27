# keras-vit

This package is a Vision Transformer (ViT) implementation based on the Keras framework. The ViT model was proposed in the paper "[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)". This package uses pre-trained weights from the ImageNet21K dataset and the ImageNet21K+ImageNet2012 dataset, which are in .npz format.

## **◈ Version Requirements and Installation**

- Python >= 3.7
- Keras >= 2.9

- To install:
  ```bash
  pip install keras-vit
  ```

## **Q1: What can you do with this package?**

- Build a pre-trained Vision Transformer (ViT) model with a standard architecture.
- Build a ViT model with custom parameters for different tasks.

## **Q2: How to build a pre-trained ViT model?**

1. **Quickly build a pre-trained ViT B16**

   ```python
   from keras_vit.vit import ViT_B16
   vit = ViT_B16()
   ```

   > *There are four configurations of pre-trained ViT models: ViT_B16, ViT_B32, ViT_L16, and ViT_L32*
   > 
   > | Configuration | Patch Size | Hidden Dim | MLP Dim | Attention Heads | Encoder Depth |
   > |:-------------:|:----------:|:----------:|:-------:|:---------------:|:-------------:|
   > | *ViT_B16*     | 16×16      | 768        | 3072    | 12              | 12            |
   > | *ViT_B32*     | 32×32      | 768        | 3072    | 12              | 12            |
   > | *ViT_L16*     | 16×16      | 1024       | 4096    | 16              | 24            |
   > | *ViT_L32*     | 32×32      | 1024       | 4096    | 16              | 24            |
   > 
   > *Pre-trained weights for the datasets "imagenet21k" and "imagenet21k+imagenet2012" have slightly different model parameters as shown below:*
   > 
   > | Weights                    | Image Size | Classes | Pre Logits | Known Labels |
   > |:--------------------------:|:----------:|:-------:|:----------:|:------------:|
   > | *imagenet21k*              | 224        | 21843   | True       | False        |
   > | *imagenet21k+imagenet2012* | 384        | 1000    | False      | True         |

2. **Build a pre-trained ViT B16 model on different datasets**

   ```python
   from keras_vit.vit import ViT_B16
   vit_1 = ViT_B16(weights="imagenet21k")
   vit_2 = ViT_B16(weights="imagenet21k+imagenet2012")
   ```

   > *The pre-trained weights (.npz) files will automatically download to: `C:\Users\user_name\.keras\weights`. If the download is interrupted, you need to delete the files in this path and re-download them, or you will encounter errors.*
   > 
   > *If the download speed is too slow, you can manually download the files ([Baidu Netdisk](https://pan.baidu.com/s/12dD74f4I2sELKbUVXaT3RA?pwd=52dl)) and place them in the above path.*

3. **Build a ViT B16 model without pre-training**

   ```python
   from keras_vit.vit import ViT_B16
   vit = ViT_B16(pre_trained=False)
   ```

4. **Build a pre-trained ViT B32 model with custom parameters**

   ```python
   from keras_vit.vit import ViT_B32
   vit = ViT_B32(
       image_size=128,
       num_classes=12,
       pre_logits=False,
       weights="imagenet21k",
   )
   ```

   > *When the parameters of the pre-trained model are changed, some layers in the model will have their parameters randomly initialized instead of loading pre-trained weights. The pre-trained weights will still load into the unchanged layers. You can check the layer loading information with the* `loading_summary()` *method.*
   
   ```python
   vit.loading_summary()
   >>
   Model: "ViT-B-32-128"
   -----------------------------------------------------------------
   Layers                             Load Weights Info
   =================================================================
   patch_embedding                    Loaded
   
   add_cls_token                      Loaded - imagenet
   
   position_embedding                 Not loaded - mismatch
   
   transformer_block_0                Loaded - imagenet
   
   transformer_block_1                Loaded - imagenet
   
   transformer_block_2                Loaded - imagenet
   
   transformer_block_3                Loaded - imagenet
   
   transformer_block_4                Loaded - imagenet
   
   transformer_block_5                Loaded - imagenet
   
   transformer_block_6                Loaded - imagenet
   
   transformer_block_7                Loaded - imagenet
   
   transformer_block_8                Loaded - imagenet
   
   transformer_block_9                Loaded - imagenet
   
   transformer_block_10               Loaded - imagenet
   
   transformer_block_11               Loaded - imagenet
   
   layer_norm                         Loaded - imagenet
   
   mlp_head                           Not loaded - mismatch
   =================================================================
   ```

## **Q3: How to build a custom ViT model?**

1. **Build a custom ViT model by instantiating the ViT class**

   ```python
   from keras_vit.vit import ViT
   vit = ViT(
       image_size=128,
       patch_size=36,
       num_classes=1,
       hidden_dim=128,
       mlp_dim=512,
       atten_heads=32,
       encoder_depth=4,
       dropout_rate=0.1,
       activation="sigmoid",
       pre_logits=True,
       include_mlp_head=True,
   )
   vit.summary()
   
   >>
   Model: "ViT-CUSTOM_SIZE-36-128"
   _________________________________________________________________
    Layer (type)                Output Shape              Param #
   =================================================================
    patch_embedding (PatchEmbed  (None, 9, 128)           497792
    ding)
   
    add_cls_token (AddCLSToken)  (None, 10, 128)          128
   
    position_embedding (AddPosi  (None, 10, 128)          1280
    tionEmbedding)
   
    transformer_block_0 (Transf  (None, 10, 128)          198272
    ormerEncoder)
   
    transformer_block_1 (Transf  (None, 10, 128)          198272
    ormerEncoder)
   
    transformer_block_2 (Transf  (None, 10, 128)          198272
    ormerEncoder)
   
    transformer_block_3 (Transf  (None, 10, 128)          198272
    ormerEncoder)
   
    layer_norm (LayerNormalizat  (None, 10, 128)          256
    ion)
   
    extract_token (Lambda)      (None, 128)               0
   
    pre_logits (Dense)          (None, 128)               16512
   
    mlp_head (Dense)            (None, 1)                 129
   
   =================================================================
   Total params: 1,309,185
   Trainable params: 1,309,185
   Non-trainable params: 0
   _________________________________________________________________
   ```

   > *Note that the* `hidden_dim` *parameter should be divisible by the* `atten_heads` *parameter. The* `image_size` *parameter should ideally be divisible by the* `patch_size` *parameter.*

2. **Load pre-trained weights into a custom ViT model**

   ```python
   from keras_vit import utils, vit
   vit_custom = vit.ViT(
       image_size=128,
       patch_size=8,
       encoder_depth=4,
   )
   utils.load_imgnet_weights(vit_custom, "ViT-B_16_imagenet21k.npz")
   vit_custom.loading_summary()
   
   >>
   Model: "ViT-CUSTOM_SIZE-8-128"
   -----------------------------------------------------------------
   Layers                             Load Weights Info
   =================================================================
   patch_embedding                    Mismatch
   
   add_cls_token                      Loaded - imagenet
   
   position_embedding                 Not loaded - mismatch
   
   transformer_block_0                Loaded - imagenet
   
   transformer_block_1                Loaded - imagenet
   
   transformer_block_2                Loaded - imagenet
   
   transformer_block_3                Loaded - imagenet
   
   layer_norm                         Loaded - imagenet
   
   pre_logits                         Loaded - imagenet
   
   mlp_head                           Not loaded - mismatch
   =================================================================
   ```

## **Q4: How to fine-tune or directly use for image classification?**

1. **Fine-tuning**

   ```python
   from keras_vit.vit import ViT_L16
   
   # Set parameters
   IMAGE_SIZE = ...
   NUM_CLASSES = ...
   ACTIVATION = ...
   ...
   
   # Build ViT
   vit = ViT_B32(
       image_size=IMAGE

_SIZE,
       num_classes=NUM_CLASSES,
       activation=ACTIVATION,
   )
   
   # Compile ViT
   vit.compile(
       optimizer=...,
       loss=...,
       metrics=...,
   )
   
   # Define train, valid, and test data
   train_generator = ...
   valid_generator = ...
   test_generator  = ...
   
   # Fine-tuning ViT
   vit.fit(
       x=train_generator,
       validation_data=valid_generator,
       steps_per_epoch=...,
       validation_steps=...,
   )
   
   # Testing
   vit.evaluate(x=test_generator, steps=...)
   ```

2. **Image classification**

   ```python
   from keras_vit import vit
   from keras_vit import utils
   
   # Get pre-trained ViT B16
   vit_model = vit.ViT_B16(weights="imagenet21k+imagenet2012")
   
   # Load an image
   img = utils.read_img("test.jpg", resize=vit_model.image_size)
   img = img.reshape((1, *vit_model.image_size, 3))
   
   # Classify
   y = vit_model.predict(img)
   classes = utils.get_imagenet2012_classes()
   print(classes[y[0].argmax()])
   ```

   > *Note that since the package currently does not include a label file for the ImageNet21k dataset, please set the weights to* `"imagenet21k+imagenet2012"` *when applying the pre-trained ViT for image classification.*
   > 
   > *For fine-tuning, both* `"imagenet21k"` *and* `"imagenet21k+imagenet2012"` *can be used.*

The project includes a script `fine_tuning_on_CIFAR10_demo.py` for fine-tuning on the [CIFAR10 dataset](https://pan.baidu.com/s/1-BCPxN57mtHh2OwQVJTOSA?pwd=52dl). Before running, unzip the dataset and place it in the `datasets` folder.
```

This Markdown content translates the original Chinese content into English while retaining the structure and technical details.
