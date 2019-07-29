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

#%% Document-Term Matrix
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer( max_features= 1000, max_df = 0.5, smooth_idf=True)# keep top 1000 terms #stop_words='english',

X = vectorizer.fit_transform(data['processed_data'])

X.shape # check shape of the document-term matrix

#%%Topic Modeling
from sklearn.decomposition import TruncatedSVD
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import sent_tokenize, word_tokenize
# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)

svd_model.fit(X)

len(svd_model.components_)

text = sent_tokenize(''.join(list(data['processed_data'].values)))

coherencemodel = CoherenceModel(model=svd_model,texts = text)

#%%
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

m1 = LdaModel(common_corpus, 3, common_dictionary)
m2 = LdaModel(common_corpus, 5, common_dictionary)

cm = CoherenceModel.for_models([m1, m2], common_dictionary, corpus=common_corpus, coherence='u_mass')












