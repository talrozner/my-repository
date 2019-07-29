#%% import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 200)
import logging
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

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
    text = re.sub(r"[^a-zA-Z0-9]",' ',text)
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
#%%
from nltk.tokenize import sent_tokenize, word_tokenize

texts = data['processed_data']
new_text = list()
for i in texts:
    #print(i)
    sentances = sent_tokenize(i)
    for k in sentances:
            new_text.append(word_tokenize(k))
texts = new_text

#%%Set up logging
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)
#logging.debug("test")

#%%Set up corpus

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


#%%Set up two topic models
goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=2)
badLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=1, num_topics=2)

#%% Using U_Mass Coherence
goodcm = CoherenceModel(model=goodLdaModel, corpus=corpus, dictionary=dictionary, coherence='u_mass')#coherence='u_mass'

badcm = CoherenceModel(model=badLdaModel, corpus=corpus, dictionary=dictionary, coherence='u_mass')

#View the pipeline parameters for one coherence model
#print(goodcm)
print(goodcm.get_coherence())

#print(badcm)
print(badcm.get_coherence())


#%% check how much topics - coherence = 'u_mass'
texts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]


results_dict = dict()
coherence_u_mass_list = list()
coherence_per_topic_list =list()
for i in range(2,10):
    print("################# topic num ############### " + str(i))
    temp_lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, iterations=1000)#
    
    temp_dcm = CoherenceModel(model=temp_lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')#'c_v'#'u_mass'
    
    coherence_score = temp_dcm.get_coherence()
    coherence_u_mass_list.append(coherence_score)
    
    results_dict[i] = {"LdaModel":temp_lda_model,"CoherenceModel":temp_dcm
                ,"coherence":coherence_score}

    coherence_per_topic_list.append(temp_dcm.get_coherence_per_topic())


num_of_topics = list(results_dict.keys())

plt.plot(num_of_topics , coherence_u_mass_list)

#%% Number of optimal Topics = 2 
temp_lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, iterations=1000)

temp_lda_model.print_topics()
#%% Using C_V coherence
'''
texts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]
goodcm = CoherenceModel(model=goodLdaModel, texts=texts, dictionary=dictionary, coherence='c_v')

badcm = CoherenceModel(model=badLdaModel, texts=texts, dictionary=dictionary, coherence='c_v')

# Pipeline parameters for C_V coherence
#print(goodcm)
print("goodcm")
print(goodcm.get_coherence())
print("badcm")
print(badcm.get_coherence())
'''
#%% check how much topics - coherence='c_v'
'''
texts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]


results_dict = dict()
coherence_list = list()
for i in range(2,3):
    print("################# topic num ############### " + str(i))
    temp_lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=i)#, iterations=100
    
    temp_dcm = CoherenceModel(model=temp_lda_model, corpus=corpus, dictionary=dictionary,texts=texts, coherence='c_v')#'c_v'#'u_mass'
    
    coherence_score = temp_dcm.get_coherence()
    coherence_list.append(coherence_score)
    
    results_dict[i] = {"LdaModel":temp_lda_model,"CoherenceModel":temp_dcm
                ,"coherence":coherence_score}

num_of_topics = list(results_dict.keys())

plt.plot(num_of_topics , coherence_list)
'''







