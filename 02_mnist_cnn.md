# MNIST handwritten digits - convolutional neural network

Study preparation. Run all cells from top to bottom. Outputs below are from a fresh Python execution, not the supplied saved outputs. Full original epoch counts and datasets are preserved. Numeric results can vary across software versions.


## Execution check

Fresh top-to-bottom Python 3.10 / TensorFlow 2.16.1 CPU execution passed on 2 November 2026. Not executed inside the hosted Google Colab service. Every code cell completed without a Python error. Warnings and limitations are discussed below. The notebook retains executed outputs.

Run in Colab: upload the `.ipynb`, connect a Python runtime and select **Runtime > Run all**. Run cells in order; do not run training cells before data/setup. Numeric results and timings may vary.


## Cell 1: Set up reproducibility

Set random seeds and CPU thread limits. No paid GPU is needed.


```python
# Reproducible CPU-friendly setup. Colab already includes the main libraries.
import os, random
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
print("TensorFlow:", tf.__version__)
print("Execution device: CPU")

```


### Actual output


```text
TensorFlow: 2.16.1
Execution device: CPU
```


## Cell 2: Import CNN tools

Conv2D learns spatial filters; MaxPooling2D downsamples; Flatten feeds the extracted features into Dense layers. SGD is the optimizer; accuracy_score independently checks predicted labels. Some original imports are unused.


```python
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
```


### Actual output


No printed output. This cell defines or updates objects used by later cells.


## Cell 3: Load MNIST

Load the full 60,000/10,000 training/test split. Images are 28 x 28 and labels are integer digits.


```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```


### Actual output


No printed output. This cell defines or updates objects used by later cells.


## Cell 4: Check training shape

Confirm the number and dimensions of training samples before reshaping.


```python
print(X_train.shape)
```


### Actual output


```text
(60000, 28, 28)
```


## Cell 5: Check raw pixel range

Raw MNIST pixels are uint8 intensities between 0 and 255. This cell shows the first image range.


```python
X_train[0].min(), X_train[0].max()
```


### Actual output


```text
(0, 255)
```


## Cell 6: Normalize

Scale both splits into [0,1]. Apply the same preprocessing at prediction time.


```python
X_train = (X_train - 0.0) / (255.0 - 0.0)
X_test = (X_test - 0.0) / (255.0 - 0.0)
X_train[0].min(), X_train[0].max()
```


### Actual output


```text
(0.0, 1.0)
```


## Cell 7: View labelled training examples

Define a helper that draws an image and its label, then plot the first 20 examples in a 4 x 5 grid.


```python
def plot_digit(image, digit, plt, i):
    plt.subplot(4, 5, i + 1)
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.title(f"Digit: {digit}")
    plt.xticks([])
    plt.yticks([])
plt.figure(figsize=(16, 10))
for i in range(20):
    plot_digit(X_train[i], y_train[i], plt, i)   
plt.show()
```


### Actual output


![Actual plotted output from cell 7](02_mnist_cnn_cell07_0.png)


## Cell 8: Add the channel dimension

Conv2D expects (samples,height,width,channels). Reshape from (N,28,28) to (N,28,28,1) because these are grayscale images.


```python
X_train = X_train.reshape((X_train.shape + (1,)))
X_test = X_test.reshape((X_test.shape + (1,)))
```


### Actual output


No printed output. This cell defines or updates objects used by later cells.


## Cell 9: Inspect labels

Show the first 20 integer labels. Sparse categorical cross-entropy accepts these directly.


```python
y_train[0:20]
```


### Actual output


```text
array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9],
dtype=uint8)
```


## Cell 10: Build the CNN

32 filters of size 3 x 3 learn local strokes. Valid convolution gives 26 x 26 x 32; 2 x 2 pooling gives 13 x 13 x 32. Flatten makes 5408 features; Dense(100) learns combinations and softmax gives 10 class probabilities.


```python
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(10, activation="softmax")
])
```


### Actual output


```text
/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
super().__init__(activity_regularizer=activity_regularizer, **kwargs)
```


## Cell 11: Compile and inspect parameters

SGD uses learning rate 0.01 and momentum 0.9. Momentum smooths gradient updates. Sparse categorical cross-entropy matches the integer labels. Total parameters: 542,230.


```python
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(
    optimizer=optimizer, 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)
model.summary()
```


### Actual output


```text
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 26, 26, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 5408)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 100)            │       540,900 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │         1,010 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 542,230 (2.07 MB)
Trainable params: 542,230 (2.07 MB)
Non-trainable params: 0 (0.00 B)
```


## Cell 12: Train

Train on all training samples for 10 epochs, batch size 32. No validation split is used in the original CNN training cell.


```python
model.fit(X_train, y_train, epochs=10, batch_size=32)
```


### Actual output


```text
[Progress output abridged; full fresh output remains in the notebook.]
━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9985 - loss: 0.0068
1839/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9985 - loss: 0.0068
1852/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9985 - loss: 0.0068
1865/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9985 - loss: 0.0068
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.9986 - loss: 0.0068
Epoch 10/10
1/1875 ━━━━━━━━━━━━━━━━━━━━ 1:14:32 2s/step - accuracy: 1.0000 - loss: 0.0043
13/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 1.0000 - loss: 0.0032
26/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9996 - loss: 0.0037
39/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9994 - loss: 0.0041
52/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9992 - loss: 0.0044
65/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9991 - loss: 0.0045
77/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9990 - loss: 0.0046
89/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9990 - loss: 0.0046
101/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9990 - loss: 0.0046
113/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9990 - loss: 0.0046
126/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9990 - loss: 0.0046
139/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9990 - loss: 0.0046
152/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9990 - loss: 0.0046
165/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9990 - loss: 0.0046
178/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9990 - loss: 0.0045
191/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9990 - loss: 0.0045
204/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0045
217/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0046
229/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0047
241/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0047
253/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0048
266/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0048
279/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0048
291/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0049
303/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0049
315/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0049
328/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0050
341/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0050
354/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0050
367/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0050
380/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0050
393/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0050
406/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0050
418/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9991 - loss: 0.0050
430/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0050
442/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0050
455/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0050
467/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0050
480/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0050
493/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0050
506/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0050
516/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0051
526/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0051
536/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0051
546/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0051
557/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0051
569/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0051
581/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0051
593/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0051
605/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0051
617/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9991 - loss: 0.0051
629/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9990 - loss: 0.0051
641/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9990 - loss: 0.0051
653/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9990 - loss: 0.0051
665/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9990 - loss: 0.0051
677/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9990 - loss: 0.0051
689/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9990 - loss: 0.0051
701/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
712/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
723/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
733/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
744/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
756/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
768/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
780/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
793/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
806/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
818/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
830/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
843/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
856/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
869/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
882/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
895/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
907/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
919/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
932/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9991 - loss: 0.0050
945/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0050
958/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0050
971/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
984/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
997/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1010/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1022/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1035/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1047/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1060/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1072/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1085/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1098/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1111/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1124/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1137/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1150/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1163/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1176/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1189/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1202/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1215/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1228/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1241/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1254/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1267/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1280/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1292/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1304/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1317/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1330/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1343/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1356/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1368/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1381/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1394/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1406/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1419/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1431/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1444/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1457/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1470/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1483/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1495/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1507/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1520/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1529/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1541/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1553/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1565/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1577/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1589/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1601/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1613/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1626/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1639/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1652/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1664/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1676/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1689/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1702/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0048
1714/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1726/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1738/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1750/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1761/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1773/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1786/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1799/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1812/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1825/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1838/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1851/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1864/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9991 - loss: 0.0049
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.9991 - loss: 0.0049
```


## Cell 13: Predict random examples

Choose 20 random test images. model.predict returns probabilities; argmax returns the predicted digit. The titles in this plot are predictions, not ground-truth labels.


```python
plt.figure(figsize=(16, 10))
for i in range(20):
    image = random.choice(X_test).squeeze()
    digit = np.argmax(model.predict(image.reshape((1, 28, 28, 1)))[0], axis=-1)
    plot_digit(image, digit, plt, i)
plt.show()
```


### Actual output


![Actual plotted output from cell 13](02_mnist_cnn_cell13_40.png)


```text
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
```


## Cell 14: Calculate test accuracy

Predict all test labels, then compare them with y_test using accuracy_score. This is the fraction correct, not the training accuracy.


```python
predictions = np.argmax(model.predict(X_test), axis=-1)
accuracy_score(y_test, predictions)
```


### Actual output


```text
1/313 ━━━━━━━━━━━━━━━━━━━━ 12s 39ms/step
34/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
65/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
97/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
129/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
162/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
197/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
232/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
267/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
302/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
0.9889
```


## Cell 15: Show one test image

Select another test image. squeeze removes the singleton channel axis for a clear grayscale display.


```python
n=random.randint(0,9999)
plt.imshow(X_test[n].squeeze(), cmap="gray")
plt.show()
```


### Actual output


![Actual plotted output from cell 15](02_mnist_cnn_cell15_0.png)


## Cell 16: Print its predicted digit

Use the softmax argmax at index n to print the digit shown above. Predicting the entire test set again is redundant but preserves the original practical flow.


```python
predicted_value=model.predict(X_test)
print("Handwritten number in the image is= %d" %np.argmax(predicted_value[n]))
```


### Actual output


```text
1/313 ━━━━━━━━━━━━━━━━━━━━ 3s 12ms/step
34/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
66/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
101/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
136/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
171/313 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step
204/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
238/313 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step
272/313 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step
305/313 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step
313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
Handwritten number in the image is= 3
```


## Cell 17: Evaluate test loss and accuracy

model.evaluate computes sparse cross-entropy and accuracy. The fresh accuracy agrees with the independent sklearn calculation. Any old numeric comments in the source are historical, not this run.


```python
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0]) #Test loss: 0.0296396646054
print('Test accuracy:', score[1])
```


### Actual output


```text
Test loss: 0.0443047396838665
Test accuracy: 0.9889000058174133
```
