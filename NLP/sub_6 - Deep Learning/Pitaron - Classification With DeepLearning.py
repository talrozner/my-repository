import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#file_path = r"D:\DS\NLP\Seminar\Code Example\Targil3\DataForTargil3.csv"
file_path = r"D:\DS\NLP\Seminar\Code Example\Targil2\Text parsing and Exploratory Data Analysis.csv"
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
max_features = 6000
maxlen = 130


x_words = feature_engineering(data,'Processed_Reviews',label_column,max_features,maxlen)

x = x_words
y = data[label_column]

# Learning from words , pos , dep
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)





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
model.add(Embedding(max_features, embed_size))
#model.add(Embedding(max_features, embed_size,input_length = maxlen))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

batch_size = 100
epochs = 3
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)


y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With One-Hot Vextor of Words:",metrics.f1_score(np.array(y_test).reshape(y_pred.shape[0],1), y_pred, average='weighted'))


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


#%% Deep learning with TF-IDF

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

batch_size = 100
epochs = 3
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)


y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With TF-IDF Vextor of Words:",metrics.f1_score(np.array(y_test).reshape(y_pred.shape[0],1), y_pred, average='weighted'))

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

#%% Deep learning with CountVectorizer

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

batch_size = 100
epochs = 10#3

history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# The returned "history" object holds a record
# of the loss values and metric values during training
print('\nhistory dict:', history.history)

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss','loss'])

y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With CountVectorizer Vextor of Words:",metrics.f1_score(np.array(y_test).reshape(y_pred.shape[0],1), y_pred, average='weighted'))





























