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

# Learning from words , pos , dep
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Model Generation Using RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0).fit(x_train, y_train)
predicted= clf.predict(x_test)
print("RandomForestClassifier Accuracy With pos and dep:",metrics.f1_score(y_test, predicted))




#%% CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 6000) 

# The input to fit_transform should be a list of strings.
train_data_features = vectorizer.fit_transform(data['Processed_Reviews'])

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

#splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_data_features,data["sentiment"],test_size=0.3,random_state=0)

################

clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0)
clf.fit(x_train, y_train)  

predicted= clf.predict(x_test)
print("RandomForest Accuracy With CountVectorizer of Words:",metrics.f1_score(y_test, predicted))
#%% tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()

x_words = v.fit_transform(data['Processed_Reviews'])
x_words = x_words.toarray()

x_pos = v.fit_transform(data['pos tag'])
x_pos = x_pos.toarray()

x_pos_dependency = v.fit_transform(data['pos dependency'])
x_pos_dependency = x_pos_dependency.toarray()

x_dep_dependency = v.fit_transform(data['dep dependency'])
x_dep_dependency = x_dep_dependency.toarray()

x = np.append(x_words,x_pos,axis=1)
x = np.append(x,x_pos_dependency,axis=1)
x = np.append(x,x_dep_dependency,axis=1)

y = data['sentiment']

# Learning from df-idf words , pos , dep
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Model Generation Using RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0).fit(x_train, y_train)
predicted= clf.predict(x_test)
print("RandomForestClassifier Accuracy With pos and dep:",metrics.f1_score(y_test, predicted))

#metrics.accuracy_score


#%% Word To Vec
#Input to the gensim's Word2Vec can be a list of sentences or list of words or list of list of sentences.
from gensim.models import Word2Vec
sentences = list(data['Processed_Reviews'].str.split(' '))#'Processed_Reviews'
#'Clean_Reviews'

#voc_vec = Word2Vec(sentences, min_count=1)
model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=4)
#model.build_vocab(sentences, update=False)
vectors = model.wv

vocab_list = list(vectors.vocab.keys())

'good' in vocab_list

vectors['good']

vectors.most_similar('good')

print(vectors.similarity('drink', 'alcohol'))
print(vectors.similarity('drink', 'sixteen'))

def get_related_terms(token, topn=10):
    
    for word, similarity in model.most_similar(positive=[token], topn=topn):
        print (word, round(similarity, 3))


get_related_terms('drink', topn=10)

answers = model.most_similar(positive=['sixteen','drink'], negative=['good'], topn=3)
print(answers)



#Plot Word Vectors Using PCA
from sklearn.decomposition import PCA

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.figure(p_num)
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
p_num+=1


#%% Create word DataFrame

# build a list of the terms, integer indices,
# and term counts from the food2vec model vocabulary
ordered_vocab = [(term, voc.index, voc.count) for term, voc in model.wv.vocab.items()]

# sort by the term counts, so the most common terms appear first
ordered_vocab = sorted(ordered_vocab, key=lambda k: -k[2])

# unzip the terms, integer indices, and counts into separate lists
ordered_terms, term_indices, term_counts = zip(*ordered_vocab)
# print(ordered_terms)
# create a DataFrame with the food2vec vectors as data,
# and the terms as row labels
word_vectors = pd.DataFrame(model.wv.syn0norm[term_indices, :], index=ordered_terms)

word_vectors

#%% cluster
X = model[model.wv.vocab]

from nltk.cluster import KMeansClusterer
import nltk
num_clusters=10
kclusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)

cluster_dataframe = pd.DataFrame({'cluster':assigned_clusters},index = vocab_list)

mask = cluster_dataframe['cluster']==0

cluster_dataframe[mask]

#%% deep learning
import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, SpatialDropout1D, Input, Bidirectional,Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

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

# Learning from only words
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_words, y, test_size=0.3, random_state=1)


vocab_size = x_words.shape[0]
embedding_vector_size = x_words.shape[1] 
embedding_matrix = x_words
max_len = 130

model = Sequential()

model.add(Embedding(input_dim = vocab_size, output_dim = embedding_vector_size, 
                    input_length = max_len, weights = [embedding_matrix]))
model.add(Bidirectional(LSTM(64, dropout=0.25, recurrent_dropout=0.1)))
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(x_train, y_train, epochs = 30, batch_size = 700, validation_data=(x_test, y_test),callbacks = [learning_rate_reduction])

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_test_pred = model.predict(x_test)

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_test_pred, average = 'weighted')




#%%
x_train, x_test, y_train, y_test = train_test_split(
    x_words,
    y,
    test_size=0.05,
    shuffle=True,
    random_state=42)

def build_model(embedding_matrix: np.ndarray, input_length: int):
    model = Sequential()
    model.add(Embedding(
        input_dim = embedding_matrix.shape[0],
        output_dim = embedding_matrix.shape[1], 
        input_length = input_length,
        weights = [embedding_matrix],
        trainable=False))
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

model = build_model(
    embedding_matrix=x_train,
    input_length=max_len)

model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(
    x=x_train,
    y=y_train,
    validation_split = 0.33,
    batch_size=100,
    epochs=20)


y_train_pred = model.predict_classes(y_train)
y_test_pred = model.predict_classes(x_test)

#roc_auc_score()




#%% Deep Learning Final
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
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 3
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With CountVectorizer of Words:",metrics.f1_score(np.array(y_test).reshape(5,1), y_pred, average='weighted'))





























