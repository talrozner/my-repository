#%% import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#%% 
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

def create_frequency_df(data,label_column,column_for_freq,num_of_words):
    #Frequency Distribution For Positive Reviews
    temp = data.loc[data[label_column]==1,column_for_freq]#'sentiment' , 'Processed_Reviews'
    temp = temp.str.cat(sep=' ')
    temp = word_tokenize(temp)
    fdist_positive = FreqDist(temp)
    #print(fdist_positive)
    #fdist_positive.most_common(2)
    frequency_word_positive = pd.DataFrame(fdist_positive.most_common(num_of_words),columns=['Word', 'Frequency'])
    
    #Frequency Distribution For Negative Reviews
    temp = data.loc[data[label_column]==0,column_for_freq]
    temp = temp.str.cat(sep=' ')
    temp = word_tokenize(temp)
    fdist_negative = FreqDist(temp)
    
    print(fdist_negative)
    
    fdist_negative.most_common(2)
    
    frequency_word_negative = pd.DataFrame(fdist_negative.most_common(num_of_words),
                        columns=['Word', 'Frequency'])
    frequency_word_positive.set_index('Word',inplace=True)
    frequency_word_negative.set_index('Word',inplace=True)
    freq_index = set(frequency_word_positive.index).union(set(frequency_word_negative.index))
    freq_index = list(freq_index)
    
    frequency_word = pd.DataFrame(index=freq_index)
    frequency_word['pos'] = frequency_word_positive
    frequency_word['neg'] = frequency_word_negative
    frequency_word.fillna(0,inplace=True)
    frequency_word.sort_values('pos',inplace=True,ascending=False)
    return(frequency_word)


#%% Frequency Distribution Plot
def plot_freq(frequency_word):
    frequency_word.plot.bar(stacked = True,fontsize = 50)
    plt.legend(['positive','negative'],fontsize = 50)
    plt.xlabel('word', fontsize=50)
    plt.ylabel('frequency', fontsize=50)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.show()
    
num_of_words = 50
frequency_word = create_frequency_df(data,'sentiment','Processed_Reviews',num_of_words)
plot_freq(frequency_word)


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

num_of_words = 50
frequency_word = create_frequency_df(data,'sentiment','pos tag',num_of_words)
plot_freq(frequency_word)




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


num_of_words = 50
frequency_word = create_frequency_df(data,'sentiment','pos dependency',num_of_words)
plot_freq(frequency_word)

num_of_words = 50
frequency_word = create_frequency_df(data,'sentiment','dep dependency',num_of_words)
plot_freq(frequency_word)

#%% feature engineering - word , pos , dep - as matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def feature_engineering(data,column,label_column,max_features,maxlen):
    #max_features = 6000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(data[column])#'Processed_Reviews'
    list_tokenized_train = tokenizer.texts_to_sequences(data[column])#'Processed_Reviews'
    
    #maxlen = 130
    x = pad_sequences(list_tokenized_train, maxlen=maxlen)
    y = data[label_column]#'sentiment'
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    return x#,reverse_word_map

#'pos tag','pos dependency', 'dep dependency'
label_column = 'sentiment'
max_features = 6000
maxlen = 130
x_words = feature_engineering(data,'Processed_Reviews',label_column,max_features,maxlen)
x_pos = feature_engineering(data,'pos tag',label_column,max_features,maxlen)
x_pos_dependency = feature_engineering(data,'pos dependency',label_column,max_features,maxlen)
x_dep_dependency = feature_engineering(data,'dep dependency',label_column,max_features,maxlen)

x = np.append(x_words,x_pos,axis=1)
x = np.append(x,x_pos_dependency,axis=1)
x = np.append(x,x_dep_dependency,axis=1)
y = data[label_column]

#%% Embedding Layer
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(data['Processed_Reviews'])

maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = data['sentiment']

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
#model.add(Bidirectional(LSTM(32, return_sequences = True)))
#model.add(GlobalMaxPool1D())
#model.add(Dense(20, activation="relu"))
#model.add(Dropout(0.05))
#model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 3
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)









