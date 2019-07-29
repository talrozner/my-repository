#%% import
import pandas as pd
import numpy as np
p_num=1

#%%load data
file_path = r"D:\DS\NLP\Bag of Words Meets Bags of Popcorn\word2vec-nlp-tutorial\labeledTrainData.tsv"
data = pd.read_table(file_path)

#%%drop column
data.drop('id',inplace=True,axis=1)

data = data[['review','sentiment']]

data = data.iloc[:100]
#%%clean data
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = re.sub(r"[,>./<)(]|['']",'',text)#, re.UNICODE#r"[,>./<')(]|[wa]|['']|['s]"
    text = text.lower()
    #text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    #text = [lemmatizer.lemmatize(token, "v") for token in text]
    #text = [word for word in text if not word in stop_words]
    #text = " ".join(text)
    return text

def lemm_text_and_remove_stop_words(text):
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

data['Clean_Reviews'] = data['review'].apply(lambda x: clean_text(x))
data['Processed_Reviews'] = data['Clean_Reviews'] .apply(lambda x: lemm_text_and_remove_stop_words(x))


#clean data - print results
data.head()
data['Processed_Reviews'].apply(lambda x: len(x.split(" "))).mean()

#%% Frequency Distribution For Positive Reviews
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
temp = data.loc[data['sentiment']==1,'Processed_Reviews']
temp = temp.str.cat(sep=' ')
temp = word_tokenize(temp)
fdist_positive = FreqDist(temp)

print(fdist_positive)

fdist_positive.most_common(2)

frequency_word_positive = pd.DataFrame(fdist_positive.most_common(50),columns=['Word', 'Frequency'])
#!!!!!!!!!!!!!!!!!

#%% Frequency Distribution For Negative Reviews
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
temp = data.loc[data['sentiment']==0,'Processed_Reviews']
temp = temp.str.cat(sep=' ')
temp = word_tokenize(temp)
fdist_negative = FreqDist(temp)

print(fdist_negative)

fdist_negative.most_common(2)

frequency_word_negative = pd.DataFrame(fdist_negative.most_common(50),
                    columns=['Word', 'Frequency'])
#!!!!!!!!!!!!!!!!!

#%% Frequency Distribution Plot Preprocess
frequency_word_positive.set_index('Word',inplace=True)
frequency_word_negative.set_index('Word',inplace=True)
freq_index = set(frequency_word_positive.index).union(set(frequency_word_negative.index))
freq_index = list(freq_index)

frequency_word = pd.DataFrame(index=freq_index)
frequency_word['pos'] = frequency_word_positive
frequency_word['neg'] = frequency_word_negative
frequency_word.fillna(0,inplace=True)
frequency_word.sort_values('pos',inplace=True,ascending=False)

#%% Frequency Distribution Plot

import matplotlib.pyplot as plt
frequency_word.plot.bar(stacked = True,fontsize = 50)
plt.legend(['positive','negative'],fontsize = 50)
plt.xlabel('word', fontsize=50)
plt.ylabel('frequency', fontsize=50)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.show()
p_num+=1


#%% POS Tagging
from nltk import pos_tag
sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=word_tokenize(sent)
pos_tag(tokens)

def create_pos_tag(sent):
    tokens=word_tokenize(sent)
    pos = pos_tag(tokens)
    pos_sent = ["".join(i[1]) for i in pos]
    str_pos_sent = ' '.join(pos_sent)
    return(str_pos_sent)

data['pos tag'] = data['Clean_Reviews'].apply(lambda x: create_pos_tag(x))

#%% pos_tag - Frequency Distribution For Positive Reviews
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
temp = data.loc[data['sentiment']==1,'pos tag']
temp = temp.str.cat(sep=' ')
temp = word_tokenize(temp)
fdist_positive = FreqDist(temp)

frequency_pos_positive = pd.DataFrame(fdist_positive.most_common(50),columns=['pos', 'Frequency'])
#!!!!!!!!!!!!!!!!!

#%% pos_tag - Frequency Distribution For Negative Reviews
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
temp = data.loc[data['sentiment']==0,'pos tag']
temp = temp.str.cat(sep=' ')
temp = word_tokenize(temp)
fdist_negative = FreqDist(temp)

frequency_pos_negative = pd.DataFrame(fdist_negative.most_common(50),
                    columns=['pos', 'Frequency'])
#!!!!!!!!!!!!!!!!!

#%% pos_tag - Frequency Distribution Plot Preprocess
frequency_pos_positive.set_index('pos',inplace=True)
frequency_pos_negative.set_index('pos',inplace=True)
freq_index = set(frequency_pos_positive.index).union(set(frequency_pos_negative.index))
freq_index = list(freq_index)

frequency_pos = pd.DataFrame(index=freq_index)
frequency_pos['pos'] = frequency_pos_positive
frequency_pos['neg'] = frequency_pos_negative
frequency_pos.fillna(0,inplace=True)
frequency_pos.sort_values('pos',inplace=True,ascending=False)

#%% pos_tag - Frequency Distribution Plot

import matplotlib.pyplot as plt
frequency_pos.plot.bar(stacked = True,fontsize = 50)
plt.legend(['positive','negative'],fontsize = 50)
plt.xlabel('pos', fontsize=50)
plt.ylabel('frequency', fontsize=50)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.show()
p_num+=1

#%%-- Text to Features -- Syntactic Parsing -- Dependency Trees
#Generating Dependency Trees using Stanford Core NLP
import spacy
nlp = spacy.load("en")
doc = nlp(u"Albert Einstein was born in Ulm, Germany in 1879.")
for token in doc:
    print (str(token.text),  str(token.lemma_),  str(token.pos_),  str(token.dep_))
    
#from spacy import displacy
#displacy.serve(doc, style='dep',page=False)

def pos_dependency_trees(x):
    nlp = spacy.load("en")
    doc = nlp(x)
    for token in doc:
        #print (str(token.text),  str(token.lemma_),  str(token.pos_),  str(token.dep_))
        
        pos_sent = ["".join(str(token.pos_)) for token in doc]
        str_pos_sent = ' '.join(pos_sent)
    
        #dep_sent = ["".join(str(token.dep_)) for token in doc]
        #str_dep_sent = ' '.join(dep_sent)
    return(str_pos_sent)

def dep_dependency_trees(x):
    nlp = spacy.load("en")
    doc = nlp(x)
    for token in doc:
     #   print (str(token.text),  str(token.lemma_),  str(token.pos_),  str(token.dep_))
            
        #pos_sent = ["".join(str(token.pos_)) for token in doc]
        #str_pos_sent = ' '.join(pos_sent)
    
        dep_sent = ["".join(str(token.dep_)) for token in doc]
        str_dep_sent = ' '.join(dep_sent)
    return(str_dep_sent)


data['pos dependency'] = data['Clean_Reviews'].apply(lambda x: pos_dependency_trees(x))

data['dep dependency'] = data['Clean_Reviews'].apply(lambda x: dep_dependency_trees(x))













