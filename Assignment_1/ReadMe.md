


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

## Project Structure (Example)

your_project_root/
├── main.py                 # FastAPI application (backend API)
├── streamlit_app.py        # Streamlit application (frontend UI)
├── embeddings.py           # Glove class for loading and working with embeddings
├── preprocessing.py        # Preprocessing class for NLP tasks (tokenization, etc.)
├── GLOVE_FILE_PATH         # Download from  [Stanford GloVe Project Page](https://nlp.stanford.edu/projects/glove/)
│ 
├── requirements.txt        # Python dependencies
└── README.md               # This file

## Setup and Installation

### Prerequisites
- Python 3.8+
- Conda (recommended for environment management)
- Access to pre-trained GloVe embeddings (e.g., `glove.6B.300d.txt`)

### 1. Clone the Repository
   ```bash
   git clone <your-github-repository-url>
   cd <your-project-directory-name>
   ```

### 2. Create and Activate Conda Environment
   It's highly recommended to use a virtual environment.
   ```bash
   conda create --name nlp_toolkit_env python=3.10 # Or your preferred Python version
   conda activate nlp_toolkit_env
   ```

### 3. Install Dependencies
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

   *(You should create a `requirements.txt` file. See section below)*

### 4. Download GloVe Embeddings
   - Download pre-trained GloVe vectors. For example, `glove.6B.zip` from [Stanford GloVe Project Page](https://nlp.stanford.edu/projects/glove/).
   - Extract the zip file.
   - Place the desired `.txt` file (e.g., `glove.6B.300d.txt`) into the appropriate directory.
   - **Important:** Update the `GLOVE_FILE_PATH` variable in `main.py` to point to the correct location of your GloVe `.txt` file if it's not automatically found relative to the script. By default, the script might expect it in a subdirectory like `Assignment_1.1/` or the same directory as `main.py`.

## Running the Application

The application consists of two main components that need to be run separately: the FastAPI backend and the Streamlit frontend.

### 1. Start the FastAPI Backend Server
   Open a terminal, activate your Conda environment, navigate to the project directory, and run:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   The API will be accessible at `http://localhost:8000` and interactive documentation at `http://localhost:8000/docs`.

### 2. Start the Streamlit Web Application
   Open a *new* terminal, activate your Conda environment, navigate to the project directory, and run:
   ```bash
   streamlit run streamlit_app.py --server.port 8501
   ```
   The web app will typically open automatically in your browser at `http://localhost:8501`.

## API Endpoints

The FastAPI backend exposes the following main endpoints (check `http://localhost:8000/docs` for full details):

**Preprocessing Endpoints (from Assignment 1):**
- `POST /tokenize`: Tokenizes input text.
- `POST /lemmatize`: Lemmatizes input text.
- `POST /stem`: Stems input text.
- `POST /pos_tag`: Performs Part-of-Speech tagging.
- `POST /ner`: Performs Named Entity Recognition.
- `POST /process/all`: Performs all above preprocessing tasks.

**Embedding Endpoints (from Assignment 2):**
- `GET /embedding/{word_input}`: Retrieves the GloVe embedding for the given `word_input`.
- `GET /nearest-neighbors/{query_word_input}?top_n=5`: Finds the `top_n` nearest neighbors for `query_word_input` from the GloVe vocabulary.

## Usage (Streamlit App)

1.  Navigate to the Streamlit app URL (e.g., `http://localhost:8501`).
2.  **For Text Preprocessing:**
    - Enter text into the "Input Text for Preprocessing" area.
    - Click "Process Text".
    - View the tokenized, lemmatized, stemmed, POS-tagged, and NER results.
3.  **For Word Embeddings & Neighbors:**
    - Enter a single word into the "Enter a word..." input field.
    - Optionally, adjust the "Number of nearest neighbors (Top N)".
    - Click "Get Embedding & Find Neighbors".
    - View the embedding information and the list of semantically similar words.


