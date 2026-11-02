# 04_cbow - exact original source

These are the original code cells in the original order, including empty cells and original comments. No fixes or setup cells have been inserted. Source text is checked against the supplied notebook.

## Run status and setup

Outputs are cleared, not fabricated. These exact originals have NOT been freshly execution-verified. Previous successful runs and actual outputs belong to the separate revised root notebooks. In Colab the usual scientific libraries are provided. For the fifth practical the original Windows paths and missing local datasets still need separate setup; installing a package alone cannot fix those paths.

## Cell-wise notes

### Original code cell 1

This original single code cell imports tools, tokenizes the original corpus, concatenates sentences, creates window=1 context-target pairs, trains the embedding/average/softmax CBOW model for 200 epochs, then predicts a word from context. Windows may cross sentence boundaries. reverse_word[predicted_id] has no guard for the padding ID 0; the revised version has a guard.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense
from tensorflow.keras.preprocessing.text import Tokenizer

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

# D. OUTPUT / PREDICTION
reverse_word = {v: k for k, v in word_index.items()}

context = ["love", "learning"]
input_data = np.array([[word_index[w] for w in context]])

prediction = model.predict(input_data, verbose=0)
predicted_id = np.argmax(prediction)

print("\nContext Words:", context)
print("Predicted Word:", reverse_word[predicted_id])

```

Output: cleared. No fresh-run result is claimed for this original code.

### Original code cell 2

Empty original code cell, retained.

```python

```

Output: cleared. No fresh-run result is claimed for this original code.
