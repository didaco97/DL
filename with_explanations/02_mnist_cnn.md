# 02_mnist_cnn - exact original source

These are the original code cells in the original order, including empty cells and original comments. No fixes or setup cells have been inserted. Source text is checked against the supplied notebook.

## Run status and setup

Outputs are cleared, not fabricated. These exact originals have NOT been freshly execution-verified. Previous successful runs and actual outputs belong to the separate revised root notebooks. In Colab the usual scientific libraries are provided. For the fifth practical the original Windows paths and missing local datasets still need separate setup; installing a package alone cannot fix those paths.

## Cell-wise notes

### Original code cell 1

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

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 2

Load the full 60,000/10,000 training/test split. Images are 28 x 28 and labels are integer digits.

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 3

Confirm the number and dimensions of training samples before reshaping.

```python
print(X_train.shape)
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 4

Raw MNIST pixels are uint8 intensities between 0 and 255. This cell shows the first image range.

```python
X_train[0].min(), X_train[0].max()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 5

Scale both splits into [0,1]. Apply the same preprocessing at prediction time.

```python
X_train = (X_train - 0.0) / (255.0 - 0.0)
X_test = (X_test - 0.0) / (255.0 - 0.0)
X_train[0].min(), X_train[0].max()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 6

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

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 7

Conv2D expects (samples,height,width,channels). Reshape from (N,28,28) to (N,28,28,1) because these are grayscale images.

```python
X_train = X_train.reshape((X_train.shape + (1,)))
X_test = X_test.reshape((X_test.shape + (1,)))
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 8

Show the first 20 integer labels. Sparse categorical cross-entropy accepts these directly.

```python
y_train[0:20]
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 9

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

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 10

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

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 11

Train on all training samples for 10 epochs, batch size 32. No validation split is used in the original CNN training cell.

```python
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 12

Choose 20 random test images. model.predict returns probabilities; argmax returns the predicted digit. The titles in this plot are predictions, not ground-truth labels.

```python
plt.figure(figsize=(16, 10))
for i in range(20):
    image = random.choice(X_test).squeeze()
    digit = np.argmax(model.predict(image.reshape((1, 28, 28, 1)))[0], axis=-1)
    plot_digit(image, digit, plt, i)
plt.show()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 13

Predict all test labels, then compare them with y_test using accuracy_score. This is the fraction correct, not the training accuracy.

```python
predictions = np.argmax(model.predict(X_test), axis=-1)
accuracy_score(y_test, predictions)
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 14

Select another test image. squeeze removes the singleton channel axis for a clear grayscale display.

```python
n=random.randint(0,9999)
plt.imshow(X_test[n])
plt.show()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 15

Use the softmax argmax at index n to print the digit shown above. Predicting the entire test set again is redundant but preserves the original practical flow.

```python
predicted_value=model.predict(X_test)
print("Handwritten number in the image is= %d" %np.argmax(predicted_value[n]))
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 16

model.evaluate computes sparse cross-entropy and accuracy. The fresh accuracy agrees with the independent sklearn calculation. Any old numeric comments in the source are historical, not this run.

```python
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0]) #Test loss: 0.0296396646054
print('Test accuracy:', score[1])
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 17

Historical source comment reporting an earlier run. It is not a result measured during this task.

```python
#The implemented CNN model is giving Loss=0.04624301567673683  and 
#accuracy: 0.9872000217437744 for test mnist dataset
```

Output: cleared. No fresh-run result is claimed for this original code.
