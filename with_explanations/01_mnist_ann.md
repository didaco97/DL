# 01_mnist_ann - exact original source

These are the original code cells in the original order, including empty cells and original comments. No fixes or setup cells have been inserted. Source text is checked against the supplied notebook.

## Run status and setup

Outputs are cleared, not fabricated. These exact originals have NOT been freshly execution-verified. Previous successful runs and actual outputs belong to the separate revised root notebooks. In Colab the usual scientific libraries are provided. For the fifth practical the original Windows paths and missing local datasets still need separate setup; installing a package alone cannot fix those paths.

## Cell-wise notes

### Original code cell 1

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

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 2

Load 60,000 training images and 10,000 test images. Each image is 28 x 28 grayscale; the integer label is the digit 0 to 9. The first run downloads the dataset.

```python
#import dataset and split into train and test data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 3

Display the second training image. Looking at examples helps confirm that the data loaded correctly.

```python
plt.matshow(x_train[1])
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 4

Original visualization uses negation on uint8 pixels. It can wrap values; a safe 255 - image visualization exists only in the separate revised version.

```python
plt.imshow(-x_train[0], cmap="gray")
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 5

Divide pixel values by 255, changing the range from 0..255 to 0..1. This puts input values on a smaller scale for gradient-based training.

```python
x_train = x_train / 255
x_test = x_test / 255
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 6

Flatten converts 28 x 28 into 784 inputs. Dense(128, relu) learns nonlinear features. Dense(10, softmax) outputs one probability per digit. Parameters: 784*128+128 + 128*10+10 = 101,770.

```python
model = keras.Sequential([
keras.layers.Flatten(input_shape=(28, 28)),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])

model.summary()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 7

SGD updates weights using gradients. Sparse categorical cross-entropy matches integer class labels; do not one-hot encode labels unless you change the loss. Accuracy is the fraction of correct class predictions.

```python
model.compile(optimizer="sgd",
loss="sparse_categorical_crossentropy",
metrics=['accuracy'])
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 8

Each epoch makes a pass through all training samples. Keras uses batch size 32 by default. This original practical uses the official test set as validation_data: no test gradients are taken, but repeated inspection makes it unsuitable as a pristine final holdout. For a serious experiment use a separate validation split.

```python
history=model.fit(x_train,
y_train,validation_data=(x_test,y_test),epochs=10)
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 9

Evaluate loss and accuracy on the test set. The fresh run prints rounded loss and accuracy; these are measured results, not promised values for every run.

```python
test_loss,test_acc=model.evaluate(x_test,y_test)
print("Loss=%.3f" %test_loss)
print("Accuracy=%.3f" %test_acc)
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 10

Pick a random test index and display its image. The fixed seed makes this example more repeatable.

```python
n=random.randint(0,9999)
plt.imshow(x_test[n])
plt.show()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 11

Displays the entire training array. This is the exact source, not the compact shape display in the revised version.

```python
x_train
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 12

Displays the entire test array. This can make a large notebook output.

```python
x_test
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 13

Predict a 10-element softmax vector for each test image and show the selected image. Position 0 represents digit 0, position 1 digit 1, and so on. The largest probability gives the predicted digit (np.argmax(predicted_value[n])).

```python
predicted_value=model.predict(x_test)
plt.imshow(x_test[n])
plt.show()

print(predicted_value[n])
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 14

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

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 15

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

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 16

Empty original code cell, retained.

```python

```

Output: cleared. No fresh-run result is claimed for this original code.
