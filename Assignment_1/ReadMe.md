


# NLP Toolkit: Preprocessing & Word Embeddings API with Streamlit UI

This project provides a comprehensive toolkit for common Natural Language Processing (NLP) tasks, including text preprocessing and word embedding analysis. It features a robust FastAPI backend and an interactive Streamlit web application for demonstration.

## Features

**1. Text Preprocessing (via API and Web App):**
   - Tokenization
   - Lemmatization
   - Stemming (Porter Stemmer)
   - Part-of-Speech (POS) Tagging
   - Named Entity Recognition (NER)
   - A comprehensive endpoint to perform all preprocessing tasks at once.

**2. Word Embeddings & Analysis (via API and Web App):**
   - Utilizes pre-trained GloVe word embeddings (e.g., `glove.6B.300d.txt`).
   - API endpoint to retrieve the embedding vector for any input word.
   - API endpoint to find the top N nearest neighbors (semantically similar words) for an input word from the GloVe vocabulary, based on cosine similarity.

**3. Interactive Web Application:**
   - Built with Streamlit for a user-friendly interface.
   - Allows users to input text for preprocessing and view results.
   - Allows users to input a word to see its GloVe embedding details and find its nearest neighbors.




