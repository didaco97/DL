# 05_transfer_learning_detection - exact original source

These are the original code cells in the original order, including empty cells and original comments. No fixes or setup cells have been inserted. Source text is checked against the supplied notebook.

## Run status and setup

Outputs are cleared, not fabricated. These exact originals have NOT been freshly execution-verified. Previous successful runs and actual outputs belong to the separate revised root notebooks. In Colab the usual scientific libraries are provided. For the fifth practical the original Windows paths and missing local datasets still need separate setup; installing a package alone cannot fix those paths.

## Cell-wise notes

### Original code cell 1

Original MobileNetV2 classification workflow: imports, Windows-local directory paths, image augmentation/rescaling, pretrained frozen base, custom head, 5 epochs, evaluation, fine-tuning for 5 epochs and plots. The referenced private dataset is not included. Paths and dataset must be supplied separately; exact source is not claimed runnable in Colab unchanged. The original uses generic 1/255 rescaling rather than MobileNetV2-specific preprocessing; corrected preprocessing and BatchNorm handling exist only in the revised version.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# =========================================================
# 1. PARAMETERS
# =========================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

BASE_PATH = r"C:\Users\Sairam\OneDrive\Desktop\IT\dataset"

TRAIN_PATH = BASE_PATH + r"\train"
VALIDATION_PATH = BASE_PATH + r"\validation"
TEST_PATH = BASE_PATH + r"\test"


# =========================================================
# 2. DATA PREPROCESSING
# =========================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# =========================================================
# 3. LOAD TRAINING DATA
# =========================================================

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)


# =========================================================
# 4. LOAD VALIDATION DATA
# =========================================================

validation_data = validation_datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# =========================================================
# 5. LOAD TEST DATA
# =========================================================

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# =========================================================
# 6. NUMBER OF CLASSES
# =========================================================

num_classes = train_data.num_classes

print("\nNumber of classes:", num_classes)
print("Class names:", train_data.class_indices)


# =========================================================
# 7. LOAD PRE-TRAINED MOBILENETV2
# =========================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)


# =========================================================
# 8. FREEZE PRE-TRAINED CNN LAYERS
# =========================================================

base_model.trainable = False


# =========================================================
# 9. ADD CUSTOM CLASSIFIER
# =========================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

output = Dense(
    num_classes,
    activation="softmax"
)(x)


# =========================================================
# 10. CREATE FINAL MODEL
# =========================================================

model = Model(
    inputs=base_model.input,
    outputs=output
)


# =========================================================
# 11. COMPILE MODEL
# =========================================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 12. DISPLAY MODEL
# =========================================================

model.summary()


# =========================================================
# 13. TRAIN CUSTOM CLASSIFIER
# =========================================================

history1 = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5
)


# =========================================================
# 14. EVALUATE BEFORE FINE-TUNING
# =========================================================

test_loss1, test_accuracy1 = model.evaluate(test_data)

print("\nTest Loss Before Fine-Tuning:",
      test_loss1)

print("Test Accuracy Before Fine-Tuning:",
      test_accuracy1)


# =========================================================
# 15. FINE-TUNING
# =========================================================

base_model.trainable = True


# Freeze all layers except last 20 layers

for layer in base_model.layers[:-20]:
    layer.trainable = False


# =========================================================
# 16. RECOMPILE WITH SMALL LEARNING RATE
# =========================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 17. FINE-TUNE MODEL
# =========================================================

history2 = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5
)


# =========================================================
# 18. FINAL EVALUATION
# =========================================================

test_loss2, test_accuracy2 = model.evaluate(test_data)

print("\nFinal Test Loss:",
      test_loss2)

print("Final Test Accuracy:",
      test_accuracy2)


# =========================================================
# 19. COMBINE TRAINING HISTORY
# =========================================================

accuracy = history1.history["accuracy"] + \
           history2.history["accuracy"]

val_accuracy = history1.history["val_accuracy"] + \
               history2.history["val_accuracy"]

loss = history1.history["loss"] + \
       history2.history["loss"]

val_loss = history1.history["val_loss"] + \
           history2.history["val_loss"]


# =========================================================
# 20. PLOT ACCURACY
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy,
    label="Validation Accuracy"
)

plt.axvline(
    x=4,
    linestyle="--",
    label="Fine-Tuning Starts"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "MobileNetV2 Transfer Learning Accuracy"
)

plt.legend()
plt.show()


# =========================================================
# 21. PLOT LOSS
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.axvline(
    x=4,
    linestyle="--",
    label="Fine-Tuning Starts"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "MobileNetV2 Transfer Learning Loss"
)

plt.legend()
plt.show()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 2

Original YOLO11n detection workflow: pretrained weights, local data.yaml, 10 training epochs at imgsz=640 and batch=8, validation and prediction on a Windows-local test.jpg. It requires Ultralytics plus a YOLO dataset/annotations and matching paths. Dataset substitutions, 320 image size and CPU settings are only in the revised version, not this exact source.

```python
# ============================================================
# OBJECT DETECTION USING TRANSFER LEARNING
# Bird, Cat and Dog Detection
# ============================================================

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2


# ------------------------------------------------------------
# 1. Load Pre-trained YOLO Model
# ------------------------------------------------------------

model = YOLO("yolo11n.pt")


# ------------------------------------------------------------
# 2. Train / Fine-Tune Model
# ------------------------------------------------------------

results = model.train(
    data=r"C:\Users\Sairam\OneDrive\Desktop\IT\dataset\data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    name="bird_cat_dog_detection"
)


# ------------------------------------------------------------
# 3. Validate Model
# ------------------------------------------------------------

metrics = model.val()

print("\nValidation completed")


# ------------------------------------------------------------
# 4. Load Test Image
# ------------------------------------------------------------

image_path = r"C:\Users\Sairam\OneDrive\Desktop\IT\dataset\test.jpg"


# ------------------------------------------------------------
# 5. Perform Object Detection
# ------------------------------------------------------------

results = model.predict(
    source=image_path,
    conf=0.25
)


# ------------------------------------------------------------
# 6. Display Detection Result
# ------------------------------------------------------------

for result in results:

    annotated_image = result.plot()

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(10, 8))

    plt.imshow(annotated_image)

    plt.axis("off")

    plt.title("Bird, Cat and Dog Object Detection")

    plt.show()
```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 3

Empty original code cell, retained.

```python

```

Output: cleared. No fresh-run result is claimed for this original code.
