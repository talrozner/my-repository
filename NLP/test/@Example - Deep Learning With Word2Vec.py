import numpy as np
import pandas as pd

file_path = r"D:\DS\NLP\Seminar\Code Example\Code By Subject\sub_6 - Deep Learning\Text parsing and Exploratory Data Analysis.csv"
data = pd.read_csv(file_path)

#%% feature engineering - One-Hot Encoding
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
    return x#,reverse_word_map

#'pos tag','pos dependency', 'dep dependency'
label_column = 'sentiment'
max_features = 230
maxlen = 130


x_words = feature_engineering(data,'Processed_Reviews',label_column,max_features,maxlen)

x = x_words
#%% Word2Vec
from gensim.models import Word2Vec
sentences = list(data['Processed_Reviews'].str.split(' '))

model = Word2Vec(sentences, size=130, window=5, min_count=3, workers=4)

embeddings_index = model.wv

#%% Create embedding_matrix
embedding_dim = 130
max_words = 230
embedding_matrix = np.zeros((max_words, embedding_dim))
p=-1
for i in embeddings_index.vocab.keys():  
    p+=1
    try:
        embedding_vector = embeddings_index.get_vector(i)
        embedding_matrix[p] = embedding_vector
        
    except:
        pass
    print(p)
    if p == max_words:
        break
y = data['sentiment']

new_x = np.dot(x,embedding_matrix.T)

# Learning from words , pos , dep
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.3, random_state=1)

#%% Deep learning with One-Hot Encoding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn import metrics
embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))#max_features, embed_size
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

batch_size = 10#100
epochs = 3
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)


y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With One-Hot Vextor of Words:",metrics.f1_score(np.array(y_test).reshape(y_pred.shape[0],1), y_pred, average='weighted'))

'''
import numpy as np
import pandas as pd

file_path = r"D:\DS\NLP\Seminar\Code Example\Code By Subject\sub_6 - Deep Learning\Text parsing and Exploratory Data Analysis.csv"
data = pd.read_csv(file_path)



#%% feature engineering - One Hot Vactor
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
    return x#,reverse_word_map


#%%tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(max_features=6000)

x_words = v.fit_transform(data['Processed_Reviews']).toarray()

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

#%% Word2Vec
from gensim.models import Word2Vec
sentences = list(data['Processed_Reviews'].str.split(' '))

model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=4)

embeddings_index = model.wv

#%% Create embedding_matrix
embedding_dim = 100
max_words = 6000
embedding_matrix = np.zeros((max_words, embedding_dim))
p=-1
for i in embeddings_index.vocab.keys():
    print(p)
    try:
        embedding_vector = embeddings_index.get_vector(i)
        embedding_matrix[p] = embedding_vector
        p+=1
    except:
        pass
#%%NN
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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 3
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

'''






















