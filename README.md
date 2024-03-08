# Understanding Word Embeddings and Word2Vec with TensorFlow

This project demonstrates the basics of word embeddings and the Word2Vec model using TensorFlow and Keras in Python. 
It illustrates how to preprocess text data, create word embeddings using the pre-trained GloVe model, and develop a simple neural network to work with word embeddings. 
The code example includes tokenizing sentences, padding sequences, loading GloVe embeddings, and creating a model with an Embedding layer.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Installation

To get started with this project, you need to have Python and TensorFlow installed on your machine. 
Additionally, you'll need to download the GloVe pre-trained word embeddings. 

Follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sorena-Dev/nderstanding-Word-Embeddings-and-Word2Vec-using-TensorFlow.git
   cd nderstanding-Word-Embeddings-and-Word2Vec-using-TensorFlow
   ```

2. **Install the requirements:**
   ```bash
   pip install tensorflow numpy
   ```

3. **Download GloVe embeddings:**
   - Visit [GloVe's website](https://nlp.stanford.edu/projects/glove/) and download the `glove.6B.zip` file.
   - Extract the `glove.6B.100d.txt` file and place it in the project directory.

## Usage

After installation, you can run the provided script to understand how word embeddings are created and used. The script performs the following operations:

- Tokenizes sample sentences.
- Pads sequences for uniform input size.
- Loads GloVe word embeddings.
- Creates a neural network model with an Embedding layer initialized with GloVe embeddings.
- Trains the model on a simple task for demonstration purposes.
- Extracts and prints the embedding for a specific word.

To run the script, execute:

```bash
python word_embeddings_word2vec.py
```

## Features

- **Text Preprocessing:** Tokenization and sequence padding.
- **GloVe Embeddings:** Utilizes pre-trained GloVe embeddings for word representation.
- **Neural Network Model:** A simple Sequential model to demonstrate the usage of word embeddings in tasks.
- **Embedding Extraction:** Extracts and prints the embedding vector for a given word.
