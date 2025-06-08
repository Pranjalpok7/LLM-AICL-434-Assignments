import numpy as np
from numpy import ndarray
import os
from typing import List, Dict, Tuple, Set
from preprocessing import Preprocessing

class Glove:
    def __init__(self,glove_file_path:str):
        '''
        Initialize the embedding index
        '''
        self.embeddings_index = {}
        self.embedding_dim = None
        
        self._load_embeddings(glove_file_path)
    
    def _load_embeddings(self,file_path:str):
        '''
        Parses the GloVe word-embeddings file and fills
        self.embedding_index
        '''
        try:
            with open(file_path, "r",encoding='utf-8') as f:
                for line_number,line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue # Skip empty lines 
                    values = line.split()
                    word = values[0]
                    embeddings = values[1:]
                    try:
                        np_vector = np.asarray(embeddings, dtype = np.float32)
                        if self.embedding_dim is None:
                            self.embedding_dim = len(np_vector)
                        self.embeddings_index[word] = np_vector
                    except ValueError as ve:
                        print(f" Line {line_number}: Could not convert embeddings to float - {ve}")
                    
                #self.embedding_dim = len(np_vector)
        except FileNotFoundError:
            print(f"File Not Found: {file_path}") 

        


    def get_embedding(self,word:str) -> ndarray | None:
        '''
        Returns the embedding vector for a given word
        '''
        return self.embeddings_index.get(word,None)
    
    def cosine_similarity(self,vec_A:ndarray, vec_B:ndarray ) -> float:
        '''
        Returns cosine similarity between two vectors
        '''
        if vec_A is None or vec_B is None:
            return 0

        dot_product = np.dot(vec_A,vec_B)
        norm_A = np.linalg.norm(vec_A)
        norm_B = np.linalg.norm(vec_B)
        
        if norm_A == 0 or norm_B == 0:
            return 0
        else:
            similarity = (dot_product)/(norm_A*norm_B)
            return float(similarity)
    
    def get_nearest_neighbors(self, word: str, top_n: int) -> List[Tuple[str, float]]:
        similarity_score = {}
        processed_word = word.lower() # Process input word once

        #Check for word existence and get its embedding
        input_embedding = self.get_embedding(processed_word)
        if input_embedding is None:
            print(f"Debug: Word '{processed_word}' not found by get_embedding.") # Optional debug
            return []

        
        for candidate_word, candidate_embedding in self.embeddings_index.items():
            if candidate_word == processed_word: # Compare with the processed_word
                continue

            score = self.cosine_similarity(input_embedding, candidate_embedding)
            similarity_score[candidate_word] = score
        
        nearest_neighbors = sorted(similarity_score.items(), key=lambda item_tuple: item_tuple[1], reverse=True)
        return nearest_neighbors[:top_n]
        

  













if __name__ == "__main__":
    glove_file_name = "glove.6B.300d.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    glove_file_to_test = os.path.join(script_dir,glove_file_name)

    nlp_preprocessor = Preprocessing()

    if not os.path.exists(glove_file_to_test):
        print(f"ERROR: GloVe file not found at {glove_file_to_test}")
    else:
        print(f"Attempting to load embeddings from: {glove_file_to_test}")
        my_glove_model = Glove(glove_file_path=glove_file_to_test)

        if my_glove_model.embeddings_index:
            print(f"Loaded {len(my_glove_model.embeddings_index)} vectors") 
             
            print(f"Embeddings dimensions: {my_glove_model.embedding_dim}")

            words_to_try = ["hello", "world", "python", "qwetry123nonexistent"] 
             
            for w in words_to_try:
                emb = my_glove_model.get_embedding(w)
                if emb is not None:
                    print(f"Embedding for '{w}': shape {emb.shape}, first 3: {emb[:3]}")

                else:
                    print(f"Embedding for '{w}': Not found")
            '''      
            ML = "Machine learning (ML) is a subset of artificial intelligence (AI) that focuses on enabling computers to learn from data without being explicitly programmed."
            tokenized_ML = nlp_preprocessor.tokenize(ML.lower())
            corpus_vocab = set(tokenized_ML)
            corpus_word_vectors = {}
            word_not_in_glove = []
            for each_word in corpus_vocab:
                embeddings = my_glove_model.get_embedding(each_word)
                if embeddings is not None:
                    corpus_word_vectors[each_word] = embeddings
                else:
                    word_not_in_glove.append(each_word)
            print(corpus_word_vectors)
            print(word_not_in_glove)
            '''

            print(my_glove_model.get_nearest_neighbors("king",10))
                
