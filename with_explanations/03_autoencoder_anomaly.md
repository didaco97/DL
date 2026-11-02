# 03_autoencoder_anomaly - exact original source

These are the original code cells in the original order, including empty cells and original comments. No fixes or setup cells have been inserted. Source text is checked against the supplied notebook.

## Run status and setup

Outputs are cleared, not fabricated. These exact originals have NOT been freshly execution-verified. Previous successful runs and actual outputs belong to the separate revised root notebooks. In Colab the usual scientific libraries are provided. For the fifth practical the original Windows paths and missing local datasets still need separate setup; installing a package alone cannot fix those paths.

## Cell-wise notes

### Original code cell 1

This original single code cell contains the whole autoencoder workflow: load MNIST, normalize/flatten, train on digit 0, encode to 64 features, decode to 784 pixels, train 20 epochs, score test zeros/ones by reconstruction MSE, set the 95th-percentile threshold using test zeros, and plot results. Test-based threshold calibration is not an unbiased independent evaluation.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)
# ------------------------------------------------
# 2. Normalize Data
# ------------------------------------------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
# Convert 28x28 images to 784-dimensional vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
# ------------------------------------------------
# 3. Select Normal Training Data
# Digit 0 = Normal
# ------------------------------------------------
normal_train = x_train[y_train == 0]
print("Normal training samples:", normal_train.shape)
# ------------------------------------------------
# 4. Select Test Data
# Digit 0 = Normal
# Digit 1 = Anomaly
# ------------------------------------------------
test_mask = (y_test == 0) | (y_test == 1)
x_test_anomaly = x_test[test_mask]
y_test_anomaly = y_test[test_mask]
# ------------------------------------------------
# 5. Encoder
# ------------------------------------------------
input_layer = Input(shape=(784,))
encoded = Dense(128, activation="relu")(input_layer)
encoded = Dense(64, activation="relu")(encoded)
# ------------------------------------------------
# 6. Decoder
# ------------------------------------------------
decoded = Dense(128, activation="relu")(encoded)
decoded = Dense(784, activation="sigmoid")(decoded)
# ------------------------------------------------
# 7. Create Autoencoder
# ------------------------------------------------
autoencoder = Model(input_layer, decoded)
# ------------------------------------------------
# 8. Compile Model
# ------------------------------------------------
autoencoder.compile(
 optimizer="adam",
 loss="mse",
 metrics=["mae"]
)
autoencoder.summary()
# ------------------------------------------------
# 9. Train Model
# ------------------------------------------------
history = autoencoder.fit(
 normal_train,
 normal_train,
 epochs=20,
 batch_size=256,
 validation_split=0.2,
 shuffle=True
)
# ------------------------------------------------
# 10. Reconstruct Test Data
# ------------------------------------------------
reconstructed = autoencoder.predict(x_test_anomaly)
# ------------------------------------------------
# 11. Calculate Reconstruction Error
# ------------------------------------------------
reconstruction_error = np.mean(
 np.square(x_test_anomaly - reconstructed),
 axis=1
)
# ------------------------------------------------
# 12. Calculate Threshold Using Normal Data
# ------------------------------------------------
normal_test = x_test[y_test == 0]
normal_reconstructed = autoencoder.predict(normal_test)
normal_error = np.mean(
 np.square(normal_test - normal_reconstructed),
 axis=1
)
threshold = np.percentile(normal_error, 95)
print("Anomaly Threshold:", threshold)
# ------------------------------------------------
# 13. Detect Anomalies
# ------------------------------------------------
predictions = reconstruction_error > threshold
print("\nFirst 20 Predictions:")
for i in range(20):
 if predictions[i]:
      print(i, "-> Anomaly")
 else:
      print(i, "-> Normal")
# ------------------------------------------------
# 14. Display Original and Reconstructed Image
# ------------------------------------------------
index = 0
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(
 x_test_anomaly[index].reshape(28, 28),
 cmap="gray"
)
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(
 reconstructed[index].reshape(28, 28),
 cmap="gray"
)
plt.title("Reconstructed")
plt.axis("off")
plt.show()
# ------------------------------------------------
# 15. Plot Reconstruction Error
# ------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(
 reconstruction_error[y_test_anomaly == 0],
 bins=50,
 alpha=0.7,
 label="Normal (0)"
)
plt.hist(
 reconstruction_error[y_test_anomaly == 1],
 bins=50,
 alpha=0.7,
 label="Anomaly (1)"
)
plt.axvline(
 threshold,
 linestyle="--",
 label="Threshold"
)
plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Samples")
plt.title("Autoencoder Anomaly Detection")
plt.legend()
plt.show()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 2

Empty original code cell, retained.

```python

```

Output: cleared. No fresh-run result is claimed for this original code.
