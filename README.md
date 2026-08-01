# Transformer-based Neural Machine Translation System using Encoder-Decoder Architecture

A complete end-to-end English-to-Marathi neural machine translation project built with TensorFlow and Keras. This project demonstrates how a Transformer-based encoder-decoder model can be trained on a small bilingual dataset, convert text into token IDs, learn contextual relationships, and generate Marathi translations for new English sentences.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python Version" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red" alt="Keras" />
  <img src="https://img.shields.io/badge/NLP-Transformer-green" alt="NLP Transformer" />
</div>

## 🌟 Project Overview

Neural machine translation is one of the most practical and powerful applications of deep learning in natural language processing. In this project, we implement a custom Transformer architecture that learns to map English sentences into Marathi sentences using:

- Text vectorization
- Token and positional embeddings
- Encoder self-attention
- Decoder masked self-attention
- Cross-attention between encoder and decoder
- Training with teacher forcing
- Inference for sentence generation

This is a lightweight but educational implementation, ideal for understanding the core mechanics behind modern translation systems.

---

## ✨ Features

- English-to-Marathi translation using Transformer architecture
- Custom token and positional embedding layer
- Encoder block with multi-head self-attention
- Decoder block with causal masking and encoder-decoder attention
- Sequence-to-sequence generation workflow
- Training pipeline with Keras `Model.fit()`
- Real-time translation function for custom input sentences
- Simple dataset designed for learning and experimentation

---

## 🏗️ Model Architecture

The project follows the standard Transformer encoder-decoder pattern:

```text
English Sentence
        |
        v
Text Vectorization
        |
        v
Token + Positional Embedding
        |
        v
Transformer Encoder
        |
        v
        +---------------------------+
        |                           |
        v                           v
Decoder Input (with start token)  Encoder Output
        |                           |
        v                           v
Token + Positional Embedding  Cross-Attention
        |
        v
Transformer Decoder
        |
        v
Dense Softmax Layer
        |
        v
Marathi Output Tokens
```

### Core Components

1. Tokenization and padding
2. Embedding layer with positional information
3. Multi-head attention for context learning
4. Feed-forward network in each encoder/decoder layer
5. Causal masking to prevent future token leakage
6. Softmax output over the Marathi vocabulary

---

## 📁 Project Structure

```text
Transformer-based-Neural-Machine-Translation-System-using-Encoder-Decoder-Architecture/
├── README.md
├── Transformer_Neural_Machine_Translation_Encoder_Decoder.py
└── requirements.txt  
```

---

## 🧠 Dataset and Training Flow

This project uses a very small bilingual dataset:

- English examples such as:
  - "i love ai"
  - "this course is good"
  - "deep learning is powerful"
- Marathi target text like:
  - "mala ai avadte"
  - "ha course changla aahe"
  - "deep learning powerful aahe"

The target sentences are prepared with special start and end tokens:

```text
start <marathi_sentence> end
```

During training, the decoder learns to predict the next Marathi word from the previous words using teacher forcing.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Transformer-based-Neural-Machine-Translation-System-using-Encoder-Decoder-Architecture.git
cd Transformer-based-Neural-Machine-Translation-System-using-Encoder-Decoder-Architecture
```

### 2. Install dependencies

```bash
pip install tensorflow
```

If you are using a virtual environment, activate it before installing the packages.

### 3. Run the project

```bash
python Transformer_Neural_Machine_Translation_Encoder_Decoder.py
```

This script performs:

- dataset creation
- tokenization
- vocabulary generation
- model construction
- training
- translation testing on sample English sentences

---

## 🏃 Training Pipeline

The script defines:

- `TokenAndPositionEmbedding` for embedding tokens and positions
- `TransformerEncoder` for reading the source sentence
- `TransformerDecoder` for generating translated tokens one by one
- A final `Dense(vocab_size, activation="softmax")` layer to produce word probabilities

The model is trained with:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

and fit on the English input and decoder inputs:

```python
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    epochs=300,
    batch_size=2,
    verbose=1
)
```

---

## 🔤 Inference and Translation

After training, the model generates translations sentence by sentence. The decoder starts with a `start` token and repeatedly predicts the next Marathi word until it reaches the `end` token or a maximum sequence length is reached.

Example inference logic:

```python
def translate_sentence(input_sentence):
    encoder_input = english_vectorizer([input_sentence])
    decoded_sentence = "start"

    for i in range(sequence_length):
        decoder_input = marathi_vectorizer([decoded_sentence])[:, :-1]
        prediction = model.predict([encoder_input, decoder_input], verbose=0)
        predicted_token_index = np.argmax(prediction[0, i, :])
        predicted_word = index_to_word.get(predicted_token_index, "")

        if predicted_word == "end" or predicted_word == "":
            break

        decoded_sentence += " " + predicted_word

    return decoded_sentence.replace("start", "").strip()
```

---

## 📊 Example Output

The script tests several sentences including:

- "i love ai"
- "i love python"
- "this course is good"
- "teacher explains well"
- "students are learning"
- "deep learning is powerful"

Possible sample output:

```text
English : i love ai
Marathi : mala ai avadte

English : this course is good
Marathi : ha course changla aahe

English : deep learning is powerful
Marathi : deep learning powerful aahe
```

---

## ✅ Why This Project Is Valuable

This project is useful because it teaches the fundamentals of real-world NLP systems:

- sequence-to-sequence modeling
- attention mechanisms
- positional encodings
- teacher forcing
- generation during inference
- practical TensorFlow/Keras implementation

It is a great stepping stone before moving to larger transformer models such as:

- mBART
- MarianMT
- T5
- BERT-based translation pipelines
- Hugging Face Transformers models

---

## 🎯 Real-World Project View

In a real production setup, this same idea would be extended with:

- much larger bilingual datasets
- subword tokenization
- better vocabulary handling
- beam search for better translation quality
- validation and checkpointing
- evaluation metrics like BLEU score
- deployment as an API or web service
- GPU acceleration for faster training

This repository is therefore not only a learning demo but also a strong foundation for building a serious translation system.

---

## 🔮 Future Improvements

Potential upgrades for a more advanced version:

- Add BLEU score evaluation
- Use subword tokenization with BPE
- Train on a larger corpus
- Add model checkpointing and early stopping
- Improve sentence generation with beam search
- Build a simple web interface for translation
- Deploy with Flask or FastAPI

---

## 🧪 Best Practices Used

- Clear separation of dataset creation and model logic
- Explicit tokenization and padding pipeline
- Attention masking for decoder correctness
- Simple reproducible training workflow
- Easy-to-understand translation function for debugging

---

## 📌 Summary

This project demonstrates how a Transformer-based encoder-decoder model can be built from scratch for machine translation. It blends theory and practical implementation, making it ideal for beginners who want to understand sequence-to-sequence learning and modern NLP architectures.

If you are learning deep learning and natural language processing, this project is an excellent hands-on example of how real translation models work under the hood.

---

## 🚀 Run It Yourself

```bash
python Transformer_Neural_Machine_Translation_Encoder_Decoder.py
```

Explore the code, modify the dataset, adjust hyperparameters, and experiment with different sentence pairs to improve translation quality.

---

## ❤️ Project Status

This project is a learning-focused, educational implementation of Transformer-based neural machine translation and is designed to build understanding before moving to large-scale production systems.

---

## 👤 Author

**Author Name:** Shubham Nanasaheb Dalvi


**Purpose:** To learn and demonstrate how an encoder-decoder Transformer can be used for Translation.

---