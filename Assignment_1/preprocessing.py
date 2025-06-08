import spacy
from nltk.stem import PorterStemmer
class Preprocessing:
    def __init__(self):
        '''
        Takes the input string to pre-process
        '''
        self.stemmer = PorterStemmer()
        self.nlp = spacy.load("en_core_web_sm")
    
    def tokenize(self,text:str):
        '''
        Tokenize the given input and returns them as a list.
        '''
        doc = self.nlp(text)
        return [token.text for token in doc]
    
    def lemmatize(self,text:str):
        '''
        Lemmatize the given input and returns them as a list
        '''
        # 1. Preprocess the entire text with spaCy 
        doc = self.nlp(text)
        # 2. Initilaize empty string. This is going to be used,
        # while appending the list of dictionary of token and stemmed version 
        # of token later.
        lemma_list = []
        for token in doc:
            lemma_list.append({"text": token.text, "lemma": token.lemma_})
        return lemma_list
    
    def stem(self,text:str):
        '''
        Perform stemming for the given input and return the stemmed
        version of text and original text in a list.
        Note: SpaCy does not support the stemming. Use NLTK for stemming purpose
        '''
        doc = self.nlp(text) 
        stemming_list = []
        # Iterate through list of strings
        for token in doc:
            # appends list of dictionary
            stemming_list.append({"text": token.text, "stem":self.stemmer.stem(token.text)})
        return stemming_list

    def pos_tagging(self,text:str):
        '''
        Perform pos_tagging for given input and returns the pos_tags
        as a list. 
        '''
        doc = self.nlp(text)
        pos_tagging_list = []
        for token in doc:
            pos_tagging_list.append({ "text":token.text, "pos_tag": token.pos_,"tag":
            token.tag_,"explanation": spacy.explain(token.tag_)})
        return pos_tagging_list
    

    def ner(self,text:str):
        doc = self.nlp(text)
        ner_list = []
        for ent in doc.ents:
            ner_list.append({"text": ent.text, "label": ent.label_, "explanation": spacy.explain(ent.label_)})
        return ner_list


if __name__ == "__main__":
    preprocessor = Preprocessing()
    sample_text = "Apple is looking at buying U.K. startup for $1 billion. This is an example of running sentences."

    print("Tokens:", preprocessor.tokenize(sample_text))
    print("\nLemmas:", preprocessor.lemmatize(sample_text))
    print("\nStems:", preprocessor.stem(sample_text))
    print("\nPOS Tags:", preprocessor.pos_tagging(sample_text))
    print("\nNER:", preprocessor.ner(sample_text)) 




