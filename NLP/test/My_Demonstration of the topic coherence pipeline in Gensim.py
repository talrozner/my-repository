#https://markroxor.github.io/gensim/static/notebooks/topic_coherence_tutorial.html#topic=0&lambda=0.87&term=

#%% import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_colwidth", 200)


#%%Load File
file_path = r"D:\DS\NLP\Seminar\Targilim\articles_4.txt"
data = pd.read_csv(file_path, sep="\n", header=None)
data.columns = ['text']

#%% Data Preprocessing
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = re.sub(r"[,>./<)(]|['']",'',text)
    text = text.lower()
    return text

def lemm_text_and_remove_stop_words(text):
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

data['clean_text'] = data['text'].apply(lambda x: clean_text(x))
data['processed_data'] = data['clean_text'] .apply(lambda x: lemm_text_and_remove_stop_words(x))


#######################################################################
#######################################################################
#%% imports
import numpy as np
import logging
try:
    import pyLDAvis.gensim
except ImportError:
    ValueError("SKIP: please install pyLDAvis")
    
import json
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.hdpmodel import HdpModel
from gensim.models.wrappers import LdaVowpalWabbit, LdaMallet
from gensim.corpora.dictionary import Dictionary
from numpy import array

#%%Set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

#%%Set up corpus
texts = [['human', 'interface', 'computer'],
         ['survey', 'user', 'computer', 'system', 'response', 'time'],
         ['eps', 'user', 'interface', 'system'],
         ['system', 'human', 'system', 'eps'],
         ['user', 'response', 'time'],
         ['trees'],
         ['graph', 'trees'],
         ['graph', 'minors', 'trees'],
         ['graph', 'minors', 'survey']]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

#%%Set up two topic models
goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=2)
badLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=1, num_topics=2)

#%% Using U_Mass Coherence
goodcm = CoherenceModel(model=goodLdaModel, corpus=corpus, dictionary=dictionary, coherence='u_mass')#coherence='u_mass'

badcm = CoherenceModel(model=badLdaModel, corpus=corpus, dictionary=dictionary, coherence='u_mass')#coherence='u_mass'

#View the pipeline parameters for one coherence model
#print(goodcm)
print(goodcm.get_coherence())

#print(badcm)
print(badcm.get_coherence())

#%% Using C_V coherence
goodcm = CoherenceModel(model=goodLdaModel, texts=texts, dictionary=dictionary, coherence='c_v')

badcm = CoherenceModel(model=badLdaModel, texts=texts, dictionary=dictionary, coherence='c_v')

#%% Pipeline parameters for C_V coherence
print(goodcm)
print(goodcm.get_coherence())
print(badcm.get_coherence())

















