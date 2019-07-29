import warnings
warnings.filterwarnings('ignore')

# Modules for data manipulation
import numpy as np
import pandas as pd
import re

# Modules for visualization
import matplotlib.pyplot as plt
import seaborn as sb

# Tools for preprocessing input data
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tools for creating ngrams and vectorizing input data
from gensim.models import Word2Vec, Phrases

# Tools for building a model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

# Tools for assessing the quality of model prediction
from sklearn.metrics import accuracy_score, confusion_matrix

#file_path = r"D:\DS\NLP\Seminar\Code Example\Targil3\DataForTargil3.csv"
file_path = r"D:\DS\NLP\Seminar\Code Example\Targil2\Text parsing and Exploratory Data Analysis.csv"
data = pd.read_csv(file_path)

#%% One Hot Encoding
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
    #reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    word_index = dict(map(reversed, tokenizer.word_index.items()))#tokenizer.word_index
    return x,y,word_index#,reverse_word_map

#'pos tag','pos dependency', 'dep dependency'
label_column = 'sentiment'
max_features = 6000
maxlen = 130


x , y ,word_index = feature_engineering(data,'Processed_Reviews',label_column,max_features,maxlen)

#%% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#%% Word2Vec
from gensim.models import Word2Vec
sentences = list(data['Processed_Reviews'].str.split(' '))

model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=4)

embeddings_index = model.wv
'''
word_index = dict({})
for idx, key in enumerate(model.wv.vocab):
    word_index[key] = model.wv[key]
    # Or word_index[key] = model.wv.get_vector(key)
    # Or word_index[key] = model.wv.word_vec(key, use_norm=False)
#embeddings_index.vocab.items()
'''
#%% Create embedding_matrix
embedding_dim = 100
max_words = 6000
embedding_matrix = np.zeros((max_words, embedding_dim))
p=-1
for i in embeddings_index.vocab.keys():
    print(p)
    if p > max_words:
        break
    try:
        embedding_vector = embeddings_index.get_vector(i)
        embedding_matrix[p] = embedding_vector
        p+=1
    except:
        pass
#%% NN with One-Hot Vector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn import metrics
#maxlen
embed_size = embedding_dim
model = Sequential()
model.add(Embedding(max_features, embed_size,input_length=maxlen))
model.add(Bidirectional(LSTM(32, return_sequences = True)))#32
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))#20
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])#'adam'

batch_size = 100
epochs = 3
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With One-Hot Vector of Words:",metrics.f1_score(np.array(y_test).reshape(y_pred.shape[0],1), y_pred, average='weighted'))



#%%tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(max_features=130)#6000

x_words = v.fit_transform(data['Processed_Reviews']).toarray()
x = x_words
#x = pad_sequences(x_words, maxlen=maxlen)
y = data[label_column]

# Learning from df-idf words
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

embed_size = embedding_dim
model = Sequential()
model.add(Embedding(max_features, embed_size,input_length=maxlen))
model.add(Bidirectional(LSTM(32, return_sequences = True)))#32
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))#20
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])#'adam'

batch_size = 100
epochs = 3
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With TF-IDF of Words:",metrics.f1_score(np.array(y_test).reshape(y_pred.shape[0],1), y_pred, average='weighted'))


#%% CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 130) #6000

# The input to fit_transform should be a list of strings.
train_data_features = vectorizer.fit_transform(data['Processed_Reviews'])

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
train_data_features = pad_sequences(train_data_features, maxlen=maxlen)
#splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_data_features,data["sentiment"],test_size=0.3,random_state=0)


embed_size = embedding_dim
model = Sequential()
model.add(Embedding(max_features, embed_size,input_length=maxlen))
model.add(Bidirectional(LSTM(32, return_sequences = True)))#32
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))#20
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])#'adam'

batch_size = 100
epochs = 3
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With CountVector of Words:",metrics.f1_score(np.array(y_test).reshape(y_pred.shape[0],1), y_pred, average='weighted'))























