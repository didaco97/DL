# DL - SPPU BE Information Technology practical preparation

Five runnable Python notebooks, each paired with Markdown containing the same cell-wise code, explanations and actual outputs. Prepared and freshly executed on 2 November 2026 for study and viva preparation.

## Start here

1. Open a practical's `.md` to study the code and its explanation together.
2. Download its matching `.ipynb`, upload it to Google Colab, connect a Python runtime and select **Runtime > Run all**.
3. Run from the first cell in a fresh runtime. The saved outputs are evidence of the recorded run; run again to produce your own outputs. Internet is required for datasets and pretrained weights on first use. CPU works; no paid GPU is required.

These notebooks passed fresh top-to-bottom execution with Python 3.10 and TensorFlow CPU 2.16.1. They were **not executed inside the hosted Google Colab service**. Colab's preinstalled versions can change, so that service-specific check is not claimed. `requirements.txt` records the tested package versions for a separate Python environment; do not blindly replace all Colab packages. The fifth notebook installs its optional detection dependency when absent. If Colab asks for a restart after installation, restart, then Run all.

| Practical | Notebook | Cell-wise notes | Fresh run |
| --- | --- | --- | --- |
| Feed-forward neural network, MNIST | [01_mnist_ann.ipynb](01_mnist_ann.ipynb) | [01_mnist_ann.md](01_mnist_ann.md) | Passed, all 10 epochs |
| CNN image classification, MNIST | [02_mnist_cnn.ipynb](02_mnist_cnn.ipynb) | [02_mnist_cnn.md](02_mnist_cnn.md) | Passed, all 10 epochs |
| Autoencoder anomaly detection | [03_autoencoder_anomaly.ipynb](03_autoencoder_anomaly.ipynb) | [03_autoencoder_anomaly.md](03_autoencoder_anomaly.md) | Passed, all 20 epochs |
| CBOW word prediction | [04_cbow.ipynb](04_cbow.ipynb) | [04_cbow.md](04_cbow.md) | Passed, all 200 epochs |
| MobileNetV2 + YOLO11n transfer learning | [05_transfer_learning_detection.ipynb](05_transfer_learning_detection.ipynb) | [05_transfer_learning_detection.md](05_transfer_learning_detection.md) | Passed, 5+5 classifier epochs and 10 detector epochs |

## Recorded results and limits

- ANN: rounded test loss 0.162, accuracy 0.952. Original test-as-validation usage is preserved and explained; it is not a pristine final holdout after repeated inspection.
- CNN: test loss 0.0443047396838665, accuracy 0.9889000058174133. The separate sklearn calculation gives 0.9889.
- Autoencoder: normal digit 0, anomaly digit 1; threshold 0.028205265849828718. Original calibration on test zeros is retained and explicitly disclosed as evaluation leakage. Use a separate calibration set for rigorous evaluation.
- CBOW: context `love, learning` produced `deep`. The tiny corpus permits multiple plausible center words and is not a language-quality benchmark.
- MobileNetV2: tiny CIFAR-10 bird/cat/dog subset, 180 train / 30 validation / 30 test. Test accuracy before/after fine-tuning was 0.4333333373 / 0.4666666687. These are demonstration results, not full CIFAR-10 performance.
- YOLO: 15 bird/cat/dog-containing images from COCO128, split into 11 training and 4 validation images with no image overlap. Reported validation mAP50 was 0.8103571429 and mAP50-95 0.6417152853. Class coverage is sparse/uneven and the validation set is also used for model selection. There is no independent detection test result. The selected validation-image inference retained **zero boxes at confidence 0.25**; that honest output is included.

Results are observations from one seeded run, not guaranteed exact numbers on other versions or hardware. A successful Python execution is different from a strong model.

## Changes from the received files

- Recovered valid notebooks from attachments without filenames, gave descriptive filenames and split monolithic code into teachable cells without removing core stages.
- Cleared supplied saved outputs before fresh execution; included the newly produced text and plots. Added cell-wise explanations inside notebooks and Markdown.
- Added reproducibility/CPU setup. Retained full original datasets and epoch counts in the four self-contained practicals.
- Corrected uint8 inversion for one ANN visualization; replaced massive full-array displays with shape/sample displays. Corrected a CNN grayscale display with `squeeze`. Removed stale claimed CNN-result comments as evidence; any remaining numeric source comment is historical only. Keras may issue non-fatal input-shape/iterator warnings.
- Added a CBOW padding-ID guard; preserved the original concatenated corpus and documented sentence-boundary mixing.
- Replaced another computer's unavailable Windows paths in the fifth practical with automatically downloaded public data, as requested. No original custom dataset performance is claimed.
- MobileNetV2 uses its required [-1,1] preprocessing; BatchNorm is kept in inference mode while fine-tuning, and the model is recompiled when trainability changes.
- Detection uses real annotated bird/cat/dog images, not fabricated boxes. Its image size/batch changed from 640/8 to 320/4 for CPU execution. Classification batch changed from 32 to 16. Tiny selected datasets keep the practical quick; increase data for meaningful performance work.
- Original custom classifier structure, pretrained backbones, losses, optimizer approach and freeze/train/unfreeze/retrain sequence remain recognizable. Model weights and raw datasets are downloaded by the notebook rather than committed here.

## Data and reference sources

- MNIST loader: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
- CIFAR-10 dataset and classes: https://keras.io/api/datasets/cifar10/
- Original CIFAR-10 source: https://cave.cs.toronto.edu/kriz/cifar.html
- CIFAR archive mirror: https://github.com/Digital-Media/cv_data/releases/tag/cifar-10 . The notebook verifies original SHA256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` before loading. The mirror archive was also checked against original MD5 `c58f30108f718f92721af3b95e74349a` during preparation.
- COCO128 dataset/annotations: https://docs.ultralytics.com/datasets/detect/coco128 . COCO128's original same-image train/validation configuration is NOT used: the selected subset is split by image here. Respect source image/dataset and model/dependency licenses; no new blanket license is granted over third-party materials.
- MobileNetV2 preprocessing: https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/preprocess_input
- Independent student SPPU BE IT practical reference: https://github.com/RanjeetKumbhar01/SPPU-BE-IT-DL-ASSIGNMENTS . This is not an official syllabus or an assurance that a particular examiner requires these exact datasets. The user's supplied files govern the practical content.
