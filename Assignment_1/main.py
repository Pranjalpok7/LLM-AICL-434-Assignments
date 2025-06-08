from fastapi.middleware.cors import CORSMiddleware
from preprocessing import Preprocessing
from embeddings import Glove
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi import HTTPException
import logging
from numpy import ndarray

class EmbeddingOut(BaseModel):
    word: str
    embeddings: Optional[List[float]] = None # Default to none if no embedding found
    found: bool = False
class NeighborsOut(BaseModel):

    word: str
    similarity: float

class TextIn(BaseModel):
    
    text: str

class LemmaOut(BaseModel):
    text: str
    lemma: str

class StemOut(BaseModel):
    text: str
    stem: str

class PosTag(BaseModel):
    text: str
    pos_tag: str
    tag: str
    explanation: str

class NER(BaseModel):
    text: str
    label: str
    explanation: str

class AllNlpResults(BaseModel):
    original_text: str
    tokens: List[str]
    lemmas: List[LemmaOut]
    stems: List[StemOut]
    pos_tags: List[PosTag]
    named_entities: List[NER]

# Creating an instance of FastAPI
app = FastAPI (
    title= "NLP Preprocessing API",
    description= "An API to perform various MLP preprocessing tasks.",
    )

# --- CORS Configuration --- 

origins = [
    "http://localhost",
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
# -------- END -------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

preprocessor = Preprocessing()
glove_model = Glove("/home/pranjal/LLM/Assignment_1/glove.6B.300d.txt")

# Check if API is running. 

@app.get("/")

def read_root():
    return {"message": "Welcome to NLP pre-processing API"}

@app.post("/tokenize",response_model = List[str])
def api_tokenize(payload: TextIn):
    '''
    Tokenize the given text and return it as a list. 
    '''

    input_string = payload.text.strip() 
    
    if not input_string: 
        raise HTTPException(
            status_code = 400, 
            detail = "Input text cannot be empty or just whitespace."
        )
    try:
        logger.info(f"Tokenizing text: '{input_string[:5]}....'")
        tokenized_text = preprocessor.tokenize(input_string)
        return tokenized_text
    except Exception as e:
        logger.error(f"Error during tokenization for input"
        f"'{input_string[:5]}...': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail= f"An unexpected error occured during tokenization."
            f"Please try again later or contact support if the issue persists."
        )


   

@app.post("/lemmatize",response_model = List[LemmaOut])
def api_lemmatize(payload:TextIn):
    '''
    Lemmatize the given text and return it as a list of dictionary
    '''
    input_string = payload.text.strip() 
    if not input_string: 
        raise HTTPException(
            status_code = 400, 
            detail = "Input text cannot be empty or just whitespace."
        )
    try:
        logger.info(f"Tokenizing text: '{input_string[:5]}....'")
        lemmatize_text = preprocessor.lemmatize(input_string)
        return lemmatize_text
    except Exception as e:
        logger.error(f"Error during lemmatization for input"
        f"'{input_string[:5]}...': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail= f"An unexpected error occured during lemmatization."
            
        )


@app.post("/stem", response_model = List[StemOut])
def api_stemming(payload:TextIn):
    '''
    Stems the payload(input data) and returns it as list of dictionary
    '''
    input_string = payload.text.strip() 
    
    if not input_string: 
        raise HTTPException(
            status_code = 400, 
            detail = "Input text cannot be empty or just whitespace."
        )
    try:
        logger.info(f"Tokenizing text: '{input_string[:5]}....'")
        stemmed_text = preprocessor.stem(input_string)
        return stemmed_text
    except Exception as e:
        logger.error(f"Error during stemming for input"
        f"'{input_string[:5]}...': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail= f"An unexpected error occured during stemming."
        
        )


@app.post("/pos_tag",response_model=List[PosTag])
def api_postag(payload:TextIn):
    '''
    Returns the tags of payload as List of dictionary
    '''
    input_string = payload.text.strip() 
    if not input_string: 
        raise HTTPException(
            status_code = 400, 
            detail = "Input text cannot be empty or just whitespace."
        )
    try:
        logger.info(f"Tokenizing text: '{input_string[:5]}....'")
        postag_list = preprocessor.pos_tagging(input_string)
        return postag_list
    except Exception as e:
        logger.error(f"Error during postaggin for input"
        f"'{input_string[:5]}...': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail= f"An unexpected error occured during postagging."
            f"Please try again later or contact support if the issue persists."
        )


@app.post("/ner",response_model=List[NER])
def api_ner(payload:TextIn):
    '''
    Returns the Name ,Enities for given payload(text)
    '''
    input_string = payload.text.strip() 
    if not input_string: 
        raise HTTPException(
            status_code = 400, 
            detail = "Input text cannot be empty or just whitespace."
        )
    try:
        logger.info(f"Tokenizing text: '{input_string[:5]}....'")
        ner = preprocessor.ner(input_string)
        return ner
    except Exception as e:
        logger.error(f"Error during Name Entity Recognition for input"
        f"'{input_string[:5]}...': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail= f"An unexpected error occured during Name Entity Recognition."
            f"Please try again later or contact support if the issue persists."
        )


@app.post("/process/all",response_model= AllNlpResults)
def api_process_all(payload:TextIn):

    '''
    Performs all the NLP preprocessing
    '''
    input_string = payload.text.strip()

    if not input_string: 
        raise HTTPException(
            status_code = 400, 
            detail = "Input text cannot be empty or just whitespace."
        )

    try:
        logger.info(f"Preprocessing text: '{input_string[:5]}....'")
        tokens = preprocessor.tokenize(input_string)
        lemmas = preprocessor.lemmatize(input_string)
        stems = preprocessor.stem(input_string)
        pos_tags = preprocessor.pos_tagging(input_string)
        named_entities = preprocessor.ner(input_string)

        return AllNlpResults(
            original_text=input_string,
            tokens = tokens,
            lemmas = lemmas,
            stems = stems,
            pos_tags = pos_tags,
            named_entities = named_entities
        )
        
    except Exception as e:
        logger.error(f"Error during preprocessing for input"
        f"'{input_string[:5]}...': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail= f"An unexpected error occured during preprocessing."
            f"Please try again later or contact support if the issue persists."
        )  

@app.get("/embedding/{word_input}", response_model=EmbeddingOut) 
def api_get_single_embedding(word_input: str):
    """
    Returns the GloVe embedding for a single input word.
    """
    if glove_model is None or not glove_model.embeddings_index: 
        raise HTTPException(status_code=503, detail="GloVe model is not available.")

    processed_word = word_input.strip().lower()
    if not processed_word:
        raise HTTPException(status_code=400, detail="Input word cannot be empty.")

    embedding_vector_np = glove_model.get_embedding(processed_word) 

    if embedding_vector_np is not None:
        return EmbeddingOut(word=processed_word, embedding=embedding_vector_np.tolist(), found=True)
    else:
        return EmbeddingOut(word=processed_word, found=False)

@app.get("/nearest-neighbors/{query_word_input}", response_model=List[NeighborsOut]) 
def api_get_all_nearest_neighbors(query_word_input: str, top_n: int = 5):
    """
    Returns the top_n nearest neighbors for a given word from the GloVe vocabulary.
    """
    if glove_model is None or not glove_model.embeddings_index: 
        raise HTTPException(status_code=503, detail="GloVe model is not available.")

    processed_word = query_word_input.strip().lower() 
    if not processed_word:
        raise HTTPException(status_code=400, detail="Input word cannot be empty.")

    try:
        
        neighbor_tuples = glove_model.get_nearest_neighbors(
            processed_word, 
            top_n
        )

        response_payload = []
        for neighbor_word, sim_score_val in neighbor_tuples: 
            response_payload.append(NeighborsOut(word=neighbor_word, similarity=sim_score_val))
        return response_payload

    except Exception as e:
        logger.error(f"Error getting nearest neighbors for '{processed_word}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error finding nearest neighbors.")