# Transfer learning - MobileNetV2 classification and YOLO bird/cat/dog detection

The original Windows-local dataset was unavailable. Per your instruction to find suitable data, classification uses a deterministic CIFAR-10 bird/cat/dog subset (60 train, 10 validation, 10 test images per class). Detection uses bird/cat/dog-labelled images from COCO128, with a disjoint image-level train/validation split. These tiny datasets are for runnable practical demonstrations, NOT reliable accuracy benchmarks.

The MobileNetV2 freeze, classifier training, fine-tuning and evaluation sequence and YOLO11n training/validation/prediction are preserved. Classification uses the required MobileNetV2 [-1,1] preprocessing instead of the original [0,1] scaling. Fine-tuning keeps BatchNorm layers frozen. All 5+5 classification epochs and 10 detection epochs are retained; detection image size is reduced from 640 to 320 for CPU execution.

Sources: https://keras.io/api/datasets/cifar10/ and https://docs.ultralytics.com/datasets/detect/coco128 . Independent student reference for SPPU BE IT task scope (not an official syllabus): https://github.com/RanjeetKumbhar01/SPPU-BE-IT-DL-ASSIGNMENTS .



## Execution check

Fresh top-to-bottom Python 3.10 / TensorFlow 2.16.1 CPU execution passed on 2 November 2026. Not executed inside the hosted Google Colab service. Every code cell completed without a Python error. Warnings and limitations are discussed below. The notebook retains executed outputs.

Run in Colab: upload the `.ipynb`, connect a Python runtime and select **Runtime > Run all**. Run cells in order; do not run training cells before data/setup. Numeric results and timings may vary.


## Cell 1: Check optional dependencies

Install Ultralytics only if unavailable, keeping NumPy below 2 for TensorFlow 2.16 compatibility. Colab provides TensorFlow, NumPy and Matplotlib. First-time installation and downloads require internet; restart the runtime if package changes ask for it.


```python
# Install only if imports fail. Do not reinstall Colab TensorFlow.
import importlib.util, subprocess, sys
if importlib.util.find_spec("ultralytics") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.3.40", "numpy<2", "opencv-python<4.12"])
print("Dependencies available")
```


### Actual output


```text
Dependencies available
```


## Cell 2: Set up reproducibility

Set seeds and CPU limits before loading TensorFlow. This notebook forces CPU so it does not require paid accelerators.


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


## Cell 3: Prepare public classification data

Download CIFAR-10 from an archive mirror and verify the original SHA256 before Keras loads it. Select bird/cat/dog (original IDs 2/3/5). For each class keep 60 training, 10 validation and 10 official-test images. Save folder-based PNGs for the original directory-loader approach. These are tiny selected subsets, not full-dataset accuracy results.


```python
from pathlib import Path
import shutil, zipfile
from PIL import Image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Official CIFAR-10 mirror, verified against original archive SHA256.
# Pre-cache the exact archive so Keras does not use its slow original host.
from tensorflow.keras.utils import get_file
get_file("cifar-10-batches-py-target_archive", origin="https://github.com/Digital-Media/cv_data/releases/download/cifar-10/cifar-10-python.tar.gz", file_hash="6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce")
# Keep original official test images for test only.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
BASE_PATH = Path.cwd() / "data" / "cifar_bird_cat_dog"
for label, name in [(2,"bird"),(3,"cat"),(5,"dog")]:
    train_ids = np.flatnonzero(y_train.ravel() == label)
    test_ids = np.flatnonzero(y_test.ravel() == label)
    for split, images in [("train",x_train[train_ids[:60]]), ("validation",x_train[train_ids[60:70]]), ("test",x_test[test_ids[:10]])]:
        folder = BASE_PATH / split / name
        folder.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            Image.fromarray(image).save(folder / f"{i:04d}.png")
print("Classification dataset: 180 train, 30 validation, 30 test; bird/cat/dog")
IMG_SIZE=(128,128)
BATCH_SIZE=16
```


### Actual output


```text
Classification dataset: 180 train, 30 validation, 30 test; bird/cat/dog
```


## Cell 4: Load and preprocess images

Resize images to 128 x 128. Augment only training images; keep validation/test fixed. MobileNetV2 preprocess_input scales [0,255] to [-1,1], unlike the original generic rescale=1/255. Directory names assign bird=0, cat=1, dog=2; assert that all splits agree.


```python
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data = train_datagen.flow_from_directory(BASE_PATH/"train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", seed=42)
validation_data = eval_datagen.flow_from_directory(BASE_PATH/"validation", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)
test_data = eval_datagen.flow_from_directory(BASE_PATH/"test", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)
assert train_data.class_indices == validation_data.class_indices == test_data.class_indices
print("Classes:",train_data.class_indices)
```


### Actual output


```text
Found 180 images belonging to 3 classes.
Found 30 images belonging to 3 classes.
Found 30 images belonging to 3 classes.
Classes: {'bird': 0, 'cat': 1, 'dog': 2}
```


## Cell 5: Build the transfer-learning classifier

Load ImageNet-pretrained MobileNetV2 without its original classifier. Freeze its weights, average spatial feature maps, add Dense(128), Dropout(0.5), and a 3-class softmax. training=False keeps BatchNorm running statistics stable. Compile with Adam and categorical cross-entropy because directory loaders return one-hot labels.


```python
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128,128,3))
base_model.trainable = False
inputs = tf.keras.Input(shape=(128,128,3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation="softmax")(x)
model = Model(inputs, outputs)
model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
```


### Actual output


```text
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)      │ (None, 128, 128, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ mobilenetv2_1.00_128            │ (None, 4, 4, 1280)     │     2,257,984 │
│ (Functional)                    │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 1280)           │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │       163,968 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 3)              │           387 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 2,422,339 (9.24 MB)
Trainable params: 164,355 (642.01 KB)
Non-trainable params: 2,257,984 (8.61 MB)
```


## Cell 6: Train the new head

Train only the custom classifier for 5 epochs, then measure the test loss/accuracy before fine-tuning. The base convolutional features remain fixed.


```python
history1 = model.fit(train_data, validation_data=validation_data, epochs=5, verbose=2)
test_loss1, test_accuracy1 = model.evaluate(test_data, verbose=0)
print("Before fine-tuning:",test_loss1,test_accuracy1)
```


### Actual output


```text
Epoch 1/5
12/12 - 4s - 311ms/step - accuracy: 0.3778 - loss: 1.8816 - val_accuracy: 0.4000 - val_loss: 1.2424
Epoch 2/5
12/12 - 1s - 72ms/step - accuracy: 0.5056 - loss: 1.3269 - val_accuracy: 0.6000 - val_loss: 0.9209
Epoch 3/5
12/12 - 1s - 65ms/step - accuracy: 0.6222 - loss: 0.9309 - val_accuracy: 0.5667 - val_loss: 1.0359
Epoch 4/5
12/12 - 1s - 67ms/step - accuracy: 0.4944 - loss: 1.0061 - val_accuracy: 0.5333 - val_loss: 0.9484
Epoch 5/5
12/12 - 1s - 66ms/step - accuracy: 0.6944 - loss: 0.7696 - val_accuracy: 0.5000 - val_loss: 0.9525
Before fine-tuning: 0.9967086911201477 0.4333333373069763
```


## Cell 7: Fine-tune upper layers

Unfreeze the base model, then freeze every layer except the last 20 and keep BatchNorm layers frozen. Recompile after changing trainability, use a smaller learning rate 0.0001, and train 5 more epochs. A lower learning rate reduces damage to pretrained features.


```python
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
history2 = model.fit(train_data, validation_data=validation_data, epochs=5, verbose=2)
test_loss2, test_accuracy2 = model.evaluate(test_data, verbose=0)
print("After fine-tuning:",test_loss2,test_accuracy2)
```


### Actual output


```text
Epoch 1/5
12/12 - 4s - 353ms/step - accuracy: 0.5833 - loss: 0.8971 - val_accuracy: 0.5333 - val_loss: 0.9292
Epoch 2/5
12/12 - 1s - 79ms/step - accuracy: 0.5778 - loss: 0.8977 - val_accuracy: 0.5000 - val_loss: 0.9585
Epoch 3/5
12/12 - 1s - 75ms/step - accuracy: 0.6111 - loss: 0.7870 - val_accuracy: 0.5000 - val_loss: 0.9779
Epoch 4/5
12/12 - 1s - 111ms/step - accuracy: 0.6333 - loss: 0.7726 - val_accuracy: 0.7000 - val_loss: 0.8318
Epoch 5/5
12/12 - 1s - 79ms/step - accuracy: 0.7111 - loss: 0.7116 - val_accuracy: 0.3667 - val_loss: 1.0728
After fine-tuning: 0.9574477076530457 0.46666666865348816
```


## Cell 8: Plot both training stages

Combine the two histories for accuracy and loss plots. The vertical line separates frozen-head training from fine-tuning. Tiny validation and test sets make these curves/accuracies unstable.


```python
for metric in ["accuracy","loss"]:
    plt.figure(figsize=(8,4))
    plt.plot(history1.history[metric]+history2.history[metric],label="Training")
    plt.plot(history1.history["val_"+metric]+history2.history["val_"+metric],label="Validation")
    plt.axvline(4.5,linestyle="--",color="gray",label="Fine-tuning starts")
    plt.xlabel("Epoch (zero-indexed)"); plt.ylabel(metric)
    plt.title("MobileNetV2 "+metric); plt.legend(); plt.show()
```


### Actual output


![Actual plotted output from cell 8](https://github.com/user-attachments/assets/58f1a517-3c5e-4744-9c75-e3c8a92c51c0)


![Actual plotted output from cell 8](https://github.com/user-attachments/assets/b36057da-a632-4717-8537-71f69d2332c7)


## Cell 9: Prepare real detection annotations

Download public COCO128, verify archive paths before extraction, and retain images with bird/cat/dog bounding boxes. Remap COCO IDs 14/15/16 to 0/1/2. Deterministically hold every fourth image out: 11 train and 4 validation images, with no shared images. YAML records relative image folders and class names. Box coordinates stay in YOLO format: class, x_center, y_center, width, height, normalized to [0,1]. No independent detection test set is available; class coverage is very uneven.


```python
# Download COCO128 and filter real bounding-box labels to bird/cat/dog.
from tensorflow.keras.utils import get_file
import yaml
archive = get_file("coco128.zip", origin="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip")
raw_root = Path.cwd()/"data"/"raw_coco"
with zipfile.ZipFile(archive) as z:
    # Only extract expected dataset members; refuse path traversal.
    for member in z.infolist():
        dest=(raw_root/member.filename).resolve()
        if not dest.is_relative_to(raw_root.resolve()):
            raise ValueError("Unsafe archive member")
    z.extractall(raw_root)
source=raw_root/"coco128"
dest=Path.cwd()/"data"/"coco_bird_cat_dog"
class_map={14:0,15:1,16:2}
records=[]
for label in sorted((source/"labels"/"train2017").glob("*.txt")):
    lines=[]
    for row in label.read_text().splitlines():
        fields=row.split()
        if int(fields[0]) in class_map:
            fields[0]=str(class_map[int(fields[0])]); lines.append(" ".join(fields))
    if lines:
        records.append((label.stem,lines))
# Deterministic split: every fourth image is validation; no image is in both.
for i,(stem,lines) in enumerate(records):
    split="val" if i%4==0 else "train"
    (dest/"images"/split).mkdir(parents=True,exist_ok=True)
    (dest/"labels"/split).mkdir(parents=True,exist_ok=True)
    shutil.copy(source/"images"/"train2017"/(stem+".jpg"),dest/"images"/split/(stem+".jpg"))
    (dest/"labels"/split/(stem+".txt")).write_text("\n".join(lines)+"\n")
config={"path":str(dest.resolve()),"train":"images/train","val":"images/val","names":{0:"bird",1:"cat",2:"dog"}}
data_yaml=dest/"data.yaml"
data_yaml.write_text(yaml.safe_dump(config))
for split in ["train","val"]:
    print(split,"images:",len(list((dest/"images"/split).glob("*.jpg"))))
print("Small pipeline demonstration only; validation is not a robust benchmark.")
```


### Actual output


```text
Downloading data from https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
0/6983030 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step
6983030/6983030 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
train images: 11
val images: 4
Small pipeline demonstration only; validation is not a robust benchmark.
```


## Cell 10: Fine-tune YOLO11n

Load pretrained YOLO11n and fine-tune it for 10 epochs on the annotated subset. The 320-pixel image size and batch 4 are CPU-friendly changes from the original 640/batch 8. YOLO predicts boxes and class scores, unlike MobileNet classification. The dataset is only a pipeline demonstration.


```python
from ultralytics import YOLO
import cv2
# Explicit CPU and zero workers keep this runnable without a paid GPU.
detector=YOLO("yolo11n.pt")
results=detector.train(data=str(data_yaml),epochs=10,imgsz=320,batch=4,
    device="cpu",workers=0,name="bird_cat_dog_detection",project="runs",
    exist_ok=True,seed=42,verbose=False,plots=False)
```


### Actual output


```text
[Progress output abridged; full fresh output remains in the notebook.]
 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.18it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.13it/s]
all          4          4    0.00978          1       0.81      0.642
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
0%|          | 0/3 [00:00<?, ?it/s]
3/10         0G     0.9017      3.484      1.097          4        320:   0%|          | 0/3 [00:00<?, ?it/s]
3/10         0G     0.9017      3.484      1.097          4        320:  33%|███▎      | 1/3 [00:00<00:00,  3.22it/s]
3/10         0G     0.9509      3.399      1.199         10        320:  33%|███▎      | 1/3 [00:00<00:00,  3.22it/s]
3/10         0G     0.9509      3.399      1.199         10        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.08it/s]
3/10         0G      1.096      3.496      1.143          9        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.08it/s]
3/10         0G      1.096      3.496      1.143          9        320: 100%|██████████| 3/3 [00:00<00:00,  3.58it/s]
3/10         0G      1.096      3.496      1.143          9        320: 100%|██████████| 3/3 [00:00<00:00,  3.44it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/1 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.31it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.26it/s]
all          4          4    0.00994          1      0.806      0.636
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
0%|          | 0/3 [00:00<?, ?it/s]
4/10         0G      1.027      3.247       1.25         11        320:   0%|          | 0/3 [00:00<?, ?it/s]
4/10         0G      1.027      3.247       1.25         11        320:  33%|███▎      | 1/3 [00:00<00:00,  2.80it/s]
4/10         0G     0.8633      3.247      1.172          4        320:  33%|███▎      | 1/3 [00:00<00:00,  2.80it/s]
4/10         0G     0.8633      3.247      1.172          4        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.08it/s]
4/10         0G      1.187      3.381      1.098          9        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.08it/s]
4/10         0G      1.187      3.381      1.098          9        320: 100%|██████████| 3/3 [00:00<00:00,  3.49it/s]
4/10         0G      1.187      3.381      1.098          9        320: 100%|██████████| 3/3 [00:00<00:00,  3.33it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/1 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.03it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  7.98it/s]
all          4          4    0.00987          1       0.75      0.621
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
0%|          | 0/3 [00:00<?, ?it/s]
5/10         0G     0.3972      3.144     0.9833          4        320:   0%|          | 0/3 [00:00<?, ?it/s]
5/10         0G     0.3972      3.144     0.9833          4        320:  33%|███▎      | 1/3 [00:00<00:00,  2.97it/s]
5/10         0G      0.696      3.224      1.082         18        320:  33%|███▎      | 1/3 [00:00<00:00,  2.97it/s]
5/10         0G      0.696      3.224      1.082         18        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.04it/s]
5/10         0G     0.7923      3.386      1.069          3        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.04it/s]
5/10         0G     0.7923      3.386      1.069          3        320: 100%|██████████| 3/3 [00:00<00:00,  3.47it/s]
5/10         0G     0.7923      3.386      1.069          3        320: 100%|██████████| 3/3 [00:00<00:00,  3.33it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/1 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  7.87it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  7.82it/s]
all          4          4    0.00993          1      0.769      0.626
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
0%|          | 0/3 [00:00<?, ?it/s]
6/10         0G     0.7641      3.298     0.8413          4        320:   0%|          | 0/3 [00:00<?, ?it/s]
6/10         0G     0.7641      3.298     0.8413          4        320:  33%|███▎      | 1/3 [00:00<00:00,  2.97it/s]
6/10         0G     0.7526      3.253     0.9916         11        320:  33%|███▎      | 1/3 [00:00<00:00,  2.97it/s]
6/10         0G     0.7526      3.253     0.9916         11        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.17it/s]
6/10         0G     0.8485      3.262     0.9397          8        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.17it/s]
6/10         0G     0.8485      3.262     0.9397          8        320: 100%|██████████| 3/3 [00:00<00:00,  3.54it/s]
6/10         0G     0.8485      3.262     0.9397          8        320: 100%|██████████| 3/3 [00:00<00:00,  3.41it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/1 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.11it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.06it/s]
all          4          4     0.0101          1       0.79      0.641
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
0%|          | 0/3 [00:00<?, ?it/s]
7/10         0G      0.837       3.18     0.8524         11        320:   0%|          | 0/3 [00:00<?, ?it/s]
7/10         0G      0.837       3.18     0.8524         11        320:  33%|███▎      | 1/3 [00:00<00:00,  3.21it/s]
7/10         0G     0.9808      3.323     0.9837         11        320:  33%|███▎      | 1/3 [00:00<00:00,  3.21it/s]
7/10         0G     0.9808      3.323     0.9837         11        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.09it/s]
7/10         0G      0.868      3.158     0.9426          3        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.09it/s]
7/10         0G      0.868      3.158     0.9426          3        320: 100%|██████████| 3/3 [00:00<00:00,  3.53it/s]
7/10         0G      0.868      3.158     0.9426          3        320: 100%|██████████| 3/3 [00:00<00:00,  3.41it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/1 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.23it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.19it/s]
all          4          4     0.0104          1      0.553      0.432
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
0%|          | 0/3 [00:00<?, ?it/s]
8/10         0G     0.7323      3.306      1.103         11        320:   0%|          | 0/3 [00:00<?, ?it/s]
8/10         0G     0.7323      3.306      1.103         11        320:  33%|███▎      | 1/3 [00:00<00:00,  2.70it/s]
8/10         0G     0.8157      3.185      1.062          4        320:  33%|███▎      | 1/3 [00:00<00:00,  2.70it/s]
8/10         0G     0.8157      3.185      1.062          4        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.02it/s]
8/10         0G      1.047      3.399      1.018         10        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.02it/s]
8/10         0G      1.047      3.399      1.018         10        320: 100%|██████████| 3/3 [00:00<00:00,  3.30it/s]
8/10         0G      1.047      3.399      1.018         10        320: 100%|██████████| 3/3 [00:00<00:00,  3.18it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/1 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  7.71it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  7.67it/s]
all          4          4       0.01          1      0.569      0.444
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
0%|          | 0/3 [00:00<?, ?it/s]
9/10         0G      1.227      3.671     0.9793          9        320:   0%|          | 0/3 [00:00<?, ?it/s]
9/10         0G      1.227      3.671     0.9793          9        320:  33%|███▎      | 1/3 [00:00<00:00,  3.23it/s]
9/10         0G      1.071      3.415      1.099         11        320:  33%|███▎      | 1/3 [00:00<00:00,  3.23it/s]
9/10         0G      1.071      3.415      1.099         11        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.24it/s]
9/10         0G     0.8639      3.318       1.01          3        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.24it/s]
9/10         0G     0.8639      3.318       1.01          3        320: 100%|██████████| 3/3 [00:00<00:00,  3.65it/s]
9/10         0G     0.8639      3.318       1.01          3        320: 100%|██████████| 3/3 [00:00<00:00,  3.52it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/1 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.26it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.19it/s]
all          4          4       0.01          1      0.569      0.444
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
0%|          | 0/3 [00:00<?, ?it/s]
10/10         0G     0.3386       2.64          1          3        320:   0%|          | 0/3 [00:00<?, ?it/s]
10/10         0G     0.3386       2.64          1          3        320:  33%|███▎      | 1/3 [00:00<00:00,  3.09it/s]
10/10         0G     0.7736      3.065      1.036         18        320:  33%|███▎      | 1/3 [00:00<00:00,  3.09it/s]
10/10         0G     0.7736      3.065      1.036         18        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.04it/s]
10/10         0G     0.7878      3.045      1.012          3        320:  67%|██████▋   | 2/3 [00:00<00:00,  3.04it/s]
10/10         0G     0.7878      3.045      1.012          3        320: 100%|██████████| 3/3 [00:00<00:00,  3.49it/s]
10/10         0G     0.7878      3.045      1.012          3        320: 100%|██████████| 3/3 [00:00<00:00,  3.36it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/1 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.19it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.14it/s]
all          4          4     0.0101          1      0.532      0.409
10 epochs completed in 0.005 hours.
Optimizer stripped from runs/bird_cat_dog_detection/weights/last.pt, 5.4MB
Optimizer stripped from runs/bird_cat_dog_detection/weights/best.pt, 5.4MB
Validating runs/bird_cat_dog_detection/weights/best.pt...
Ultralytics 8.3.40 🚀 Python-3.10.12 torch-2.5.1+cpu CPU (Intel Xeon Processor)
YOLO11n summary (fused): 238 layers, 2,582,737 parameters, 0 gradients, 6.3 GFLOPs
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/1 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.50it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.45it/s]
all          4          4    0.00988          1       0.81      0.642
Speed: 0.4ms preprocess, 25.2ms inference, 0.0ms loss, 1.5ms postprocess per image
```


## Cell 11: Validate and show detection output

Validate on the four held-out images and print precision, recall, mAP@0.5 and mAP@0.5:0.95. Metrics on this tiny set are not reliable generalization claims; validation is also used to pick training checkpoints. Run prediction on one validation image at confidence 0.25. The fresh run produced 0 retained boxes on this image: the plot honestly shows that, rather than claiming successful detection. Low-confidence/no boxes is a model outcome, not a Python execution error.


```python
metrics=detector.val(data=str(data_yaml),device="cpu",workers=0,plots=False)
print("Validation metrics:",metrics.results_dict)
image_path=str(sorted((dest/"images"/"val").glob("*.jpg"))[0])
print("Demo image:",Path(image_path).name,"(a validation image, not an independent test)")
predictions=detector.predict(source=image_path,conf=0.25,device="cpu",verbose=False)
for result in predictions:
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(result.plot(),cv2.COLOR_BGR2RGB))
    plt.axis("off");plt.title("Bird/cat/dog detection - small demo dataset");plt.show()
    print("Detected boxes:",len(result.boxes))
```


### Actual output


![Actual plotted output from cell 11](https://github.com/user-attachments/assets/b7039aee-a071-44cb-af58-daac3e5c259d)


```text
Ultralytics 8.3.40 🚀 Python-3.10.12 torch-2.5.1+cpu CPU (Intel Xeon Processor)
YOLO11n summary (fused): 238 layers, 2,582,737 parameters, 0 gradients, 6.3 GFLOPs
val: Scanning /downloads/practical-prep/data/coco_bird_cat_dog/labels/val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4/4 [00:00<?, ?it/s]
val: Scanning /downloads/practical-prep/data/coco_bird_cat_dog/labels/val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4/4 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/1 [00:00<?, ?it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.76it/s]
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  8.69it/s]
all          4          4    0.00988          1       0.81      0.642
Speed: 0.4ms preprocess, 24.3ms inference, 0.0ms loss, 1.6ms postprocess per image
Validation metrics: {'metrics/precision(B)': 0.009879755709227443, 'metrics/recall(B)': 1.0, 'metrics/mAP50(B)': 0.8103571428571428, 'metrics/mAP50-95(B)': 0.6417152852916315, 'fitness': 0.6585794710481827}
Demo image: 000000000042.jpg (a validation image, not an independent test)
Detected boxes: 0
```
