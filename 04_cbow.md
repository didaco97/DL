# Word prediction - continuous bag of words (CBOW)

Study preparation. Run all cells from top to bottom. Outputs below are from a fresh Python execution, not the supplied saved outputs. Full original epoch counts and datasets are preserved. Numeric results can vary across software versions.


## Execution check

Fresh top-to-bottom Python 3.10 / TensorFlow 2.16.1 CPU execution passed on 2 November 2026. Not executed inside the hosted Google Colab service. Every code cell completed without a Python error. Warnings and limitations are discussed below. The notebook retains executed outputs.

Run in Colab: upload the `.ipynb`, connect a Python runtime and select **Runtime > Run all**. Run cells in order; do not run training cells before data/setup. Numeric results and timings may vary.


## Cell 1: Set up reproducibility

Fix random seeds and CPU settings so the small training example is repeatable.


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


## Cell 2: Import CBOW tools

Tokenizer maps words to integer IDs. Embedding learns a vector for each word. Lambda averages context vectors and Dense predicts a vocabulary distribution.


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
```


### Actual output


No printed output. This cell defines or updates objects used by later cells.


## Cell 3: Prepare text and vocabulary

Tokenizer lowercases and splits the four short lines, assigning IDs to 8 distinct words. ID 0 is reserved for padding, so vocab_size is 9. The original practical concatenates lines before windowing, so some context windows cross sentence boundaries.


```python
# A. DATA PREPARATION
text = """I love machine learning
I love deep learning
machine learning is interesting
deep learning is powerful"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

data = tokenizer.texts_to_sequences([text])[0]
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

print("Vocabulary:", word_index)
```


### Actual output


```text
Vocabulary: {'learning': 1, 'i': 2, 'love': 3, 'machine': 4, 'deep': 5, 'is': 6, 'interesting': 7, 'powerful': 8}
```


## Cell 4: Build context-target pairs

window=1 means one word on each side of a center word. X contains the two surrounding IDs and y is the one-hot vector for the center word. The 16-token corpus yields 14 pairs. This is CBOW: context predicts the center word, unlike skip-gram.


```python
# B. GENERATE TRAINING DATA
X, y = [], []
window = 1

for i in range(window, len(data) - window):
    context = data[i-window:i] + data[i+1:i+window+1]
    X.append(context)
    y.append(data[i])

X = np.array(X)
y = tf.keras.utils.to_categorical(y, vocab_size)

print("\nTraining Data:")
print("X =", X)
```


### Actual output


```text
Training Data:
X = [[2 4]
[3 1]
[4 2]
[1 3]
[2 5]
[3 1]
[5 4]
[1 1]
[4 6]
[1 7]
[6 5]
[7 1]
[5 6]
[1 8]]
```


## Cell 5: Train CBOW

Embedding outputs a length-10 vector per context word. reduce_mean averages the two vectors (a bag of words ignores order), and softmax predicts the center word. Adam minimizes categorical cross-entropy for 200 epochs on the tiny corpus. There is no held-out language evaluation here.


```python
# C. TRAIN CBOW MODEL
model = Sequential([
    Embedding(vocab_size, 10),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(vocab_size, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=200, verbose=0)
```


### Actual output


No printed output. This cell defines or updates objects used by later cells.


## Cell 6: Predict a missing word

Encode [love, learning], average its embeddings, and select the largest softmax output. The context is ambiguous in the corpus because both machine and deep can fit. The observed prediction is one trained-model outcome, not a universal language rule. A padding-ID guard prevents a KeyError if index 0 is ever selected.


```python
# D. OUTPUT / PREDICTION
reverse_word = {v: k for k, v in word_index.items()}

context = ["love", "learning"]
input_data = np.array([[word_index[w] for w in context]])

prediction = model.predict(input_data, verbose=0)
predicted_id = np.argmax(prediction)

print("\nContext Words:", context)
print("Predicted Word:", reverse_word.get(int(predicted_id), "<padding>"))
```


### Actual output


```text
Context Words: ['love', 'learning']
Predicted Word: deep
```
