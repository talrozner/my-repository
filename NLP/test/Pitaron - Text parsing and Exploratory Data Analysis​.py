import pandas as pd
import numpy as np

file_path = r"D:\DS\NLP\Seminar\Code Example\Targil2\Text parsing and Exploratory Data Analysis.csv"

data = pd.read_csv(file_path)

import spacy

def pos_dependency_trees(x,tag_type):
    nlp = spacy.load("en")
    doc = nlp(x)
    if tag_type == 'tag':
        for token in doc:
        #print (str(token.text),  str(token.lemma_),  str(token.pos_),  str(token.dep_))
            tag_sent = ["".join(str(token.tag_)) for token in doc]
            str_sent = ' '.join(tag_sent)
            
    if tag_type == 'pos':
        for token in doc:
            pos_sent = ["".join(str(token.pos_)) for token in doc]
            str_sent = ' '.join(pos_sent)
            
    if tag_type == 'dep':
        for token in doc:
            dep_sent = ["".join(str(token.dep_)) for token in doc]
            str_sent = ' '.join(dep_sent)
            
    return(str_sent)


data['tag dependency'] = data['Clean_Reviews'].apply(lambda x: pos_dependency_trees(x,'tag'))

data['pos dependency'] = data['Clean_Reviews'].apply(lambda x: pos_dependency_trees(x,'pos'))

data['dep dependency'] = data['Clean_Reviews'].apply(lambda x: pos_dependency_trees(x,'dep'))

from collections import Counter 

def count_entity(x):
    nlp = spacy.load("en")
    article = nlp(x)
    labels = [x.label_ for x in article.ents]
    entities = ' '.join(labels)
    return(entities)
    
data['Entities'] = data['Clean_Reviews'].apply(lambda x: count_entity(x))


def generate_ngrams(text, n):
    words = text.split()
    output = []  
    for i in range(len(words)-n+1):
        output.append(tuple(words[i:i+n]))       
    bigrams_list = [str(x) +'-'+ str(v) for x,v in output]
    ngrams = ' '.join(bigrams_list)
    return ngrams 
   
data['ngrams'] = data['tag dependency'].apply(lambda x: generate_ngrams(x,2))






