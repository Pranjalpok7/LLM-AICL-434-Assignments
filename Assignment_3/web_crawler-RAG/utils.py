import requests
from bs4 import BeautifulSoup
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import urllib.parse

# --- Web Crawling ---
def crawl_website(url: str, max_pages: int = 5) -> List[str]: # Reduced max_pages
    """
    Crawls a website starting from a given URL up to max_pages, extracting text content.
    """
    visited = set()
    to_visit = [url]
    docs = []
    print(f"Starting crawl from {url} (max pages: {max_pages})")
    while to_visit and len(visited) < max_pages:
        current = to_visit.pop(0)
        if current in visited:
            continue
        print(f"Crawling: {current}")
        try:
            resp = requests.get(current, timeout=10) # Increased timeout
            resp.raise_for_status() # Raise an exception for bad status codes (e.g., 404, 500)
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            docs.append(text)
            visited.add(current)
            # Add new links
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Handle relative URLs
                if not href.startswith('http'):
                    href = urllib.parse.urljoin(current, href)

                # Avoid fragments and already visited links
                if href.startswith('http') and href not in visited and "#" not in href:
                    # Optional: Add logic here to restrict crawling to a specific domain
                    # from urllib.parse import urlparse
                    # if urlparse(href).netloc == urlparse(url).netloc:
                    to_visit.append(href)

        except requests.exceptions.RequestException as e:
            print(f"Failed to crawl {current} (RequestException): {e}")
        except Exception as e:
            print(f"Failed to crawl {current} (Other Error): {e}")

    print(f"Finished crawling. Crawled {len(docs)} documents.")
    return docs

# --- Text Processing (Chunking) ---
def chunk_document(document: str, chunk_size: int = 250, chunk_overlap: int = 50) -> List[str]:
    """
    Splits a document into overlapping word-based chunks.
    """
    # Use a simple regex to split into words and punctuation as potential word boundaries
    words = re.findall(r'\w+|[.,!?;:]', document) # Include some punctuation with words
    chunks = []
    if not words:
        return []

    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if chunk: # Add non-empty chunks
           chunks.append(chunk)
    return chunks

# --- Retrieval ---
def build_retrieval_index(docs: List[str], model_name='all-MiniLM-L6-v2') -> Tuple[faiss.Index, SentenceTransformer, List[str], np.ndarray]:
    """
    Builds a FAISS index from document chunks using a Sentence Transformer model.
    Returns the index, model, list of all chunks, and embeddings.
    """
    model = SentenceTransformer(model_name)
    all_chunks = []
    print(f"Chunking {len(docs)} documents...")
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    if not all_chunks:
        print("No chunks generated from documents.")
        return None, model, [], np.array([])

    print(f"Generated {len(all_chunks)} chunks. Building index...")
    # show_progress_bar=True would normally be here, but disabling for compatibility
    embeddings = model.encode(all_chunks, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    print("Index built successfully.")
    return index, model, all_chunks, embeddings # Return all_chunks

def retrieve(query: str, index, model, all_chunks: List[str], top_k: int = 3) -> List[str]: # Reduced top_k
    """
    Retrieves top_k relevant chunks for a query using the FAISS index.
    """
    if index is None or not all_chunks:
        print("Index not built or no chunks available for retrieval.")
        return []

    print(f"Retrieving top {top_k} chunks for query: '{query}'")
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), top_k)

    retrieved_chunks = [all_chunks[i] for i in I[0]]
    print(f"Retrieved {len(retrieved_chunks)} chunks.")
    return retrieved_chunks

# --- Evaluation ---
def evaluate_rag(answers: List[str], references: List[str]) -> float:
    """
    Simple exact match evaluation with basic normalization.
    Compares generated answers to reference answers.
    """
    correct = 0
    if not answers or not references or len(answers) != len(references):
        print("Answers and references lists are not valid for evaluation.")
        return 0.0

    for a, r in zip(answers, references):
        # Basic normalization: remove non-alphanumeric chars and convert to lower case
        norm_a = re.sub(r'\W+', '', str(a)).lower().strip()
        norm_r = re.sub(r'\W+', '', str(r)).lower().strip()
        if norm_a == norm_r:
            correct += 1
    return correct / len(answers) if answers else 0.0 # Added basic normalization