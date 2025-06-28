## Assignment 3.1
This project implements a basic Retrieval-Augmented Generation (RAG) pipeline. The pipeline consists of the following steps:

1.  **Web Crawling:** A function `crawl_website` is provided to extract text content from a given URL and its linked pages up to a specified maximum number of pages. It uses `requests` and `BeautifulSoup` for this purpose.
2.  **Retrieval Indexing:** The extracted documents are used to build a retrieval index. The `build_retrieval_index` function uses a Sentence Transformer model (`all-MiniLM-L6-v2` by default) to create embeddings of the documents and then uses `faiss` to build a flat L2 index for efficient similarity search.
3.  **Document Retrieval:** Given a query, the `retrieve` function uses the built index and the Sentence Transformer model to find the most relevant documents from the corpus.
4.  **Answer Generation:** A pre-trained question-answering model (specifically `distilbert-base-uncased-distilled-squad` using the `transformers` library's pipeline) is loaded. The `generate_answer` function takes a query and the retrieved documents (concatenated into a single context) and uses the QA model to generate an answer based on the provided context.
5.  **Evaluation:** A simple evaluation function `evaluate_rag` is included, which calculates the exact match score between generated answers and reference answers.

The `rag_pipeline.ipynb` notebook demonstrates the usage of these components with an example query and evaluates the result. The `utils.py` file contains the core functions for crawling, retrieval, and evaluation. 
