# 🎉 WYN-VisionModel 🐍

This package includes a variety of powerful scripts to handle image and video processing tasks, as well as implement state-of-the-art deep learning models.

## 📚 Package Contents:

### 1. convnet.py 🧠
This script houses a convolutional neural network. It's powerful, efficient, and ready to tackle your image classification tasks!

### 2. data_process.py 🖼️
In charge of your image and video processing needs. It processes images and can convert videos into image sequences!

### 3. scan.py 🎭
An amazing script that generates masks and even crafts videos from image sequences. Let it sweep across your data and watch the magic happen!

### 4. vit.py 🕶️
Our very own packaged Vision Transformer model! This is the cutting edge of deep learning for computer vision tasks, and now it's ready to deploy!

## ⚙️ Getting Started

First things first, we need to make sure all dependencies are installed. This project has a `requirements.txt` file, which you can use to install the necessary dependencies. To install the requirements, simply run:

```bash
pip install -r requirements.txt
```

Make sure to run this command before trying to use any of the scripts, otherwise you may run into issues with missing dependencies.

Then install the `wyn-visionmodel` package:

```bash
pip install git+https://github.com/yiqiao-yin/WYN-VisionModel.git
```

## 🚀 Usage

Each script in the package can be run individually with Python. For example:

```py
from src.vision_model.vit import *

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 10
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

vit_classifier = create_vit_classifier(
    x_train=x_train,
    input_shape=(32, 32, 3),
    image_size=image_size,
    patch_size=patch_size,
    num_patches=num_patches,
    projection_dim=projection_dim,
    transformer_layers=transformer_layers,
    num_heads=num_heads,
    transformer_units=transformer_units,
    mlp_head_units=mlp_head_units,
    num_classes=len(np.unique(y_train)),
)

%%time
history, model = run_experiment(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    model=vit_classifier,
    do_ckeckpoint=False,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    batch_size=batch_size,
    num_epochs=num_epochs,
)

# Epoch 1/10
176/176 [==============================] - 1140s 6s/step - loss: 4.4782 - accuracy: 0.0442 - top-5-accuracy: 0.1577 - val_loss: 3.9452 - val_accuracy: 0.1038 - val_top-5-accuracy: 0.3088
# ...
```

Just replace `convnet.py` with the script you want to run!

## 👥 Contribution

Contributions are always welcome! If you'd like to contribute, feel free to submit a pull request.

## 🎓 License

This project is licensed under the terms of the MIT license.

---
This should provide a pretty good starting point for your README file. Don't forget to update it with more specific details about how to use each script, what types of input and output they accept, and any other details that your users might need to know.