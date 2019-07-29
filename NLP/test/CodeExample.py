#%%import
import numpy as np
import pandas as pd

#%%--Text Preprocessing -- Noise Removal

import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 

def clean_text(text):
    text = re.sub(r"[@][a-zA-Z0-9]+|[.]",'',text)
    text = text.lower()
    text = [word for word in text.split(' ') if not word in stop_words]
    text = " ".join(text)
    return text

text = r"@VirginAmerica plus you've added commercials to the experience... tacky."

new_text = clean_text(text)

print("clean text: " +"\n" + str(new_text))


#%%--Text Preprocessing ----Lexicon Normalization -- Stemming
from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word = "multiplying" 

print('\n'+"original word: "+'\n'+word+'\n'+'\n'+"stem word: "+"\n"+  stem.stem(word))

#%%--Text Preprocessing ----Lexicon Normalization -- Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

print('\n'+'lemma word with pos v:' +"\n"+  lem.lemmatize(word, "v"))

#%%--Text Preprocessing -- Object Standardization
lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love"}

def _lookup_words(input_text):
    words = input_text.split() 
    new_words = [] 
    for word in words:
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
        new_words.append(word)
        new_text = " ".join(new_words) 
    return new_text

_lookup_words("RT this is a retweeted tweet by Shivam Bansal")

























