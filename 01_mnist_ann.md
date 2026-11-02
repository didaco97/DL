# MNIST handwritten digits - feed-forward neural network

Study preparation. Run all cells from top to bottom. Outputs below are from a fresh Python execution, not the supplied saved outputs. Full original epoch counts and datasets are preserved. Numeric results can vary across software versions.


## Execution check

Fresh top-to-bottom Python 3.10 / TensorFlow 2.16.1 CPU execution passed on 2 November 2026. Not executed inside the hosted Google Colab service. Every code cell completed without a Python error. Warnings and limitations are discussed below. The notebook retains executed outputs.

Run in Colab: upload the `.ipynb`, connect a Python runtime and select **Runtime > Run all**. Run cells in order; do not run training cells before data/setup. Numeric results and timings may vary.


## Cell 1: Set up reproducibility

Set fixed seeds for Python, NumPy and TensorFlow. Limit CPU thread use so this runs without a GPU. A seed reduces randomness, but different versions/hardware may still change results.


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


## Cell 2: Import libraries

TensorFlow/Keras builds and trains the network; NumPy handles arrays; Matplotlib draws images and learning curves. Pandas is imported by the original practical but is not needed here.


```python
#importing necessary libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
%matplotlib inline
```


### Actual output


No printed output. This cell defines or updates objects used by later cells.


## Cell 3: Load MNIST

Load 60,000 training images and 10,000 test images. Each image is 28 x 28 grayscale; the integer label is the digit 0 to 9. The first run downloads the dataset.


```python
#import dataset and split into train and test data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```


### Actual output


No printed output. This cell defines or updates objects used by later cells.


## Cell 4: Inspect an image

Display the second training image. Looking at examples helps confirm that the data loaded correctly.


```python
plt.matshow(x_train[1])
```


### Actual output


![Actual plotted output from cell 4](https://github.com/user-attachments/assets/f4b3e1fd-6a5a-4156-8b30-8489b7971904)


## Cell 5: Display inverted grayscale

Subtract pixel values from 255 to invert the grayscale safely. The original negative sign on uint8 data wraps around rather than performing a clean inversion. This affects only visualization, not training.


```python
# Show an inverted grayscale image without uint8 arithmetic overflow.
plt.imshow(255 - x_train[0], cmap="gray")
plt.show()
```


### Actual output


![Actual plotted output from cell 5](https://github.com/user-attachments/assets/c5b4bcf2-21a7-4695-b53c-6349dd4bf67d)


## Cell 6: Normalize pixels

Divide pixel values by 255, changing the range from 0..255 to 0..1. This puts input values on a smaller scale for gradient-based training.


```python
x_train = x_train / 255
x_test = x_test / 255
```


### Actual output


No printed output. This cell defines or updates objects used by later cells.


## Cell 7: Build the feed-forward network

Flatten converts 28 x 28 into 784 inputs. Dense(128, relu) learns nonlinear features. Dense(10, softmax) outputs one probability per digit. Parameters: 784*128+128 + 128*10+10 = 101,770.


```python
model = keras.Sequential([
keras.layers.Flatten(input_shape=(28, 28)),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])

model.summary()
```


### Actual output


```text
/usr/local/lib/python3.10/dist-packages/keras/src/layers/reshaping/flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
super().__init__(**kwargs)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten (Flatten)               │ (None, 784)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │       100,480 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 101,770 (397.54 KB)
Trainable params: 101,770 (397.54 KB)
Non-trainable params: 0 (0.00 B)
```


## Cell 8: Compile the network

SGD updates weights using gradients. Sparse categorical cross-entropy matches integer class labels; do not one-hot encode labels unless you change the loss. Accuracy is the fraction of correct class predictions.


```python
model.compile(optimizer="sgd",
loss="sparse_categorical_crossentropy",
metrics=['accuracy'])
```


### Actual output


No printed output. This cell defines or updates objects used by later cells.


## Cell 9: Train for 10 epochs

Each epoch makes a pass through all training samples. Keras uses batch size 32 by default. This original practical uses the official test set as validation_data: no test gradients are taken, but repeated inspection makes it unsuitable as a pristine final holdout. For a serious experiment use a separate validation split.


```python
history=model.fit(x_train,
y_train,validation_data=(x_test,y_test),epochs=10)
```


### Actual output


```text
[Progress output abridged; full fresh output remains in the notebook.]
9441 - loss: 0.2043
1398/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9441 - loss: 0.2043
1441/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9441 - loss: 0.2042
1482/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9441 - loss: 0.2041
1524/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9441 - loss: 0.2040
1567/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9441 - loss: 0.2039
1610/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9441 - loss: 0.2038
1652/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9441 - loss: 0.2037
1694/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9441 - loss: 0.2036
1736/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9441 - loss: 0.2035
1777/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9442 - loss: 0.2035
1819/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9442 - loss: 0.2034
1861/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9442 - loss: 0.2033
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 1ms/step - accuracy: 0.9447 - loss: 0.1991 - val_accuracy: 0.9451 - val_loss: 0.1908
Epoch 8/10
1/1875 ━━━━━━━━━━━━━━━━━━━━ 1:19:06 3s/step - accuracy: 0.9062 - loss: 0.3401
42/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9448 - loss: 0.1988
83/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9474 - loss: 0.1966
125/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9489 - loss: 0.1914
166/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9491 - loss: 0.1904
208/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9488 - loss: 0.1904
250/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9486 - loss: 0.1906
292/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9487 - loss: 0.1904
334/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9488 - loss: 0.1903
376/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9488 - loss: 0.1903
418/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9488 - loss: 0.1904
461/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9488 - loss: 0.1908
503/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9487 - loss: 0.1910
545/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9487 - loss: 0.1912
587/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9486 - loss: 0.1912
626/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9486 - loss: 0.1913
667/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9486 - loss: 0.1914
710/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9486 - loss: 0.1914
752/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9485 - loss: 0.1913
794/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9485 - loss: 0.1914
836/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9485 - loss: 0.1914
878/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9485 - loss: 0.1913
916/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9485 - loss: 0.1912
958/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9485 - loss: 0.1911
999/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9485 - loss: 0.1910
1036/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9485 - loss: 0.1909
1073/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1908
1115/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1907
1156/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1906
1198/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1905
1240/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1905
1282/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1904
1324/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1903
1366/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1902
1408/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1902
1450/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1901
1492/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1901
1533/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1900
1574/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1899
1617/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1898
1659/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1897
1701/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1897
1742/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1896
1785/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1895
1827/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1894
1869/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9485 - loss: 0.1894
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 1ms/step - accuracy: 0.9488 - loss: 0.1857 - val_accuracy: 0.9477 - val_loss: 0.1798
Epoch 9/10
1/1875 ━━━━━━━━━━━━━━━━━━━━ 24s 13ms/step - accuracy: 0.9062 - loss: 0.3220
43/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9487 - loss: 0.1845
84/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9508 - loss: 0.1826
126/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9522 - loss: 0.1780
168/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9522 - loss: 0.1771
208/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9519 - loss: 0.1772
249/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9516 - loss: 0.1775
290/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9516 - loss: 0.1773
331/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9516 - loss: 0.1774
373/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9516 - loss: 0.1774
415/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9516 - loss: 0.1776
456/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9516 - loss: 0.1779
498/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9515 - loss: 0.1782
538/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9514 - loss: 0.1784
578/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9513 - loss: 0.1785
620/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9513 - loss: 0.1786
661/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9512 - loss: 0.1787
704/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9512 - loss: 0.1787
745/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9512 - loss: 0.1787
786/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9512 - loss: 0.1788
827/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9512 - loss: 0.1788
868/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9512 - loss: 0.1788
910/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9512 - loss: 0.1787
952/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9512 - loss: 0.1786
993/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9512 - loss: 0.1786
1034/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9512 - loss: 0.1785
1076/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1784
1118/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1783
1159/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1782
1201/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1782
1243/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1781
1285/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1781
1327/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1780
1369/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1779
1412/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1779
1454/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1779
1496/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1778
1535/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9512 - loss: 0.1777
1576/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9513 - loss: 0.1777
1618/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9513 - loss: 0.1776
1661/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9513 - loss: 0.1775
1703/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9513 - loss: 0.1775
1746/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9513 - loss: 0.1774
1788/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9513 - loss: 0.1773
1830/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9513 - loss: 0.1773
1872/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9513 - loss: 0.1772
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - accuracy: 0.9517 - loss: 0.1741 - val_accuracy: 0.9501 - val_loss: 0.1702
Epoch 10/10
1/1875 ━━━━━━━━━━━━━━━━━━━━ 24s 13ms/step - accuracy: 0.9062 - loss: 0.3061
46/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9521 - loss: 0.1716
87/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9552 - loss: 0.1700
129/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9564 - loss: 0.1660
169/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9562 - loss: 0.1654
211/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9559 - loss: 0.1656
253/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9556 - loss: 0.1660
294/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9555 - loss: 0.1659
335/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9555 - loss: 0.1660
376/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9554 - loss: 0.1661
417/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9554 - loss: 0.1663
458/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9553 - loss: 0.1666
499/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9552 - loss: 0.1669
540/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9551 - loss: 0.1672
581/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9550 - loss: 0.1673
622/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9549 - loss: 0.1674
664/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9549 - loss: 0.1676
704/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9549 - loss: 0.1676
742/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9548 - loss: 0.1676
783/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9548 - loss: 0.1677
824/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9548 - loss: 0.1678
865/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9548 - loss: 0.1677
906/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9547 - loss: 0.1677
948/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9548 - loss: 0.1677
990/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9548 - loss: 0.1676
1032/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.9548 - loss: 0.1675
1074/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9548 - loss: 0.1675
1116/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9548 - loss: 0.1674
1157/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9548 - loss: 0.1673
1199/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9548 - loss: 0.1673
1241/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1673
1283/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1672
1326/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1672
1368/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1671
1411/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1671
1454/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1671
1496/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1670
1538/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1670
1579/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1669
1620/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1668
1662/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1668
1704/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1667
1746/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1667
1788/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1666
1829/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1666
1871/1875 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9547 - loss: 0.1665
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - accuracy: 0.9549 - loss: 0.1639 - val_accuracy: 0.9516 - val_loss: 0.1617
```


## Cell 10: Evaluate

Evaluate loss and accuracy on the test set. The fresh run prints rounded loss and accuracy; these are measured results, not promised values for every run.


```python
test_loss,test_acc=model.evaluate(x_test,y_test)
print("Loss=%.3f" %test_loss)
print("Accuracy=%.3f" %test_acc)
```


### Actual output


```text
1/313 ━━━━━━━━━━━━━━━━━━━━ 4s 14ms/step - accuracy: 0.9688 - loss: 0.1676
52/313 ━━━━━━━━━━━━━━━━━━━━ 0s 985us/step - accuracy: 0.9504 - loss: 0.1650
104/313 ━━━━━━━━━━━━━━━━━━━━ 0s 979us/step - accuracy: 0.9440 - loss: 0.1901
156/313 ━━━━━━━━━━━━━━━━━━━━ 0s 976us/step - accuracy: 0.9421 - loss: 0.1969
209/313 ━━━━━━━━━━━━━━━━━━━━ 0s 972us/step - accuracy: 0.9418 - loss: 0.1972
263/313 ━━━━━━━━━━━━━━━━━━━━ 0s 964us/step - accuracy: 0.9427 - loss: 0.1934
313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 997us/step - accuracy: 0.9516 - loss: 0.1617
Loss=0.162
Accuracy=0.952
```


## Cell 11: Choose a test image

Pick a random test index and display its image. The fixed seed makes this example more repeatable.


```python
n=random.randint(0,9999)
plt.imshow(x_test[n])
plt.show()
```


### Actual output


![Actual plotted output from cell 11](https://github.com/user-attachments/assets/d2c31fb5-508b-4311-b48e-6bf3c504c144)


## Cell 12: Inspect training data compactly

Print the shape and a 5 x 5 corner of the first image instead of flooding the notebook with the full training array. This does not change the data or model.


```python
print("x_train shape:", x_train.shape)
print("First image (top-left 5x5):")
print(x_train[0, :5, :5])
```


### Actual output


```text
x_train shape: (60000, 28, 28)
First image (top-left 5x5):
[[0. 0. 0. 0. 0.]
[0. 0. 0. 0. 0.]
[0. 0. 0. 0. 0.]
[0. 0. 0. 0. 0.]
[0. 0. 0. 0. 0.]]
```


## Cell 13: Inspect test data compactly

Print the test shape and sample corner. The test data is normalized in exactly the same way as the training data.


```python
print("x_test shape:", x_test.shape)
print("First image (top-left 5x5):")
print(x_test[0, :5, :5])
```


### Actual output


```text
x_test shape: (10000, 28, 28)
First image (top-left 5x5):
[[0. 0. 0. 0. 0.]
[0. 0. 0. 0. 0.]
[0. 0. 0. 0. 0.]
[0. 0. 0. 0. 0.]
[0. 0. 0. 0. 0.]]
```


## Cell 14: Predict probabilities

Predict a 10-element softmax vector for each test image and show the selected image. Position 0 represents digit 0, position 1 digit 1, and so on. The largest probability gives the predicted digit (np.argmax(predicted_value[n])).


```python
predicted_value=model.predict(x_test)
plt.imshow(x_test[n])
plt.show()

print(predicted_value[n])
```


### Actual output


![Actual plotted output from cell 14](https://github.com/user-attachments/assets/6a31acdf-46ec-432a-a5ac-5c4848d1e184)


```text
1/313 ━━━━━━━━━━━━━━━━━━━━ 8s 27ms/step
87/313 ━━━━━━━━━━━━━━━━━━━━ 0s 587us/step
179/313 ━━━━━━━━━━━━━━━━━━━━ 0s 567us/step
259/313 ━━━━━━━━━━━━━━━━━━━━ 0s 587us/step
313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 654us/step
[3.6157016e-07 9.8982835e-01 2.3489875e-04 2.0368428e-03 4.5812190e-05
6.3676515e-04 1.7222638e-04 3.9606708e-05 6.7728753e-03 2.3233175e-04]
```


## Cell 15: Plot accuracy

Training accuracy measures the training set; validation accuracy measures the test-as-validation set used above. Compare their trends to spot a growing generalization gap.


```python
# history.history()
history.history.keys()
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```


### Actual output


![Actual plotted output from cell 15](https://github.com/user-attachments/assets/fd06ed84-58fc-472c-8726-54566e6e4f27)


## Cell 16: Plot loss

Cross-entropy measures prediction error. Lower is better. A falling training loss with rising validation loss is a warning sign of overfitting.


```python
# history.history()
history.history.keys()
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```


### Actual output


![Actual plotted output from cell 16](https://github.com/user-attachments/assets/2ca0c7b7-c37d-4a9b-80bb-759c25d2f8ae)
