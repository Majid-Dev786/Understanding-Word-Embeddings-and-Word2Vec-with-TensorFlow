import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.models import Sequential
import numpy as np

# Sample sentences
sentences = [
    'I enjoy playing football.',
    'Football is a popular sport.'
]

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Word to index mapping
word_index = tokenizer.word_index

# Padding the sequences
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Creating word embeddings
embedding_dim = 100
embeddings_index = {}
with open('glove.6B.100d.txt', 'r', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Creating the model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compiling and training the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, np.array([1, 0]), epochs=10)

# Extracting word embeddings
word = 'football'
word_embedding = model.layers[0].get_weights()[0][word_index[word]]

print(f'Word: {word}\tEmbedding: {word_embedding}')
