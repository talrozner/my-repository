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


x_one_hot = feature_engineering(data,'Processed_Reviews',label_column,max_features,maxlen)


y = data[label_column]

# Learning from words , pos , dep
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_one_hot, y, test_size=0.3, random_state=1)


#%% Deep learning with One-Hot Encoding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D,GlobalMaxPool3D,Input
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn import metrics
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
history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)


y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With One-Hot Vextor of Words:",metrics.f1_score(np.array(y_test).reshape(y_pred.shape[0],1), y_pred, average='weighted'))

# The returned "history" object holds a record
# of the loss values and metric values during training
print('\nhistory dict:', history.history)

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss','loss'])

#0.8611276373023146

#%% Improve the model - Add LSTM Layers
#0.864201667032298
embed_size = 128

model = Sequential()
model.add(Embedding(max_features, embed_size))#input dim , Output dim

model.add(Bidirectional(LSTM(128, return_sequences = True)))
model.add(Bidirectional(LSTM(64, return_sequences = True)))
model.add(Bidirectional(LSTM(32, return_sequences = True)))


model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

batch_size = 100
epochs = 3
history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)


y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With One-Hot Vextor of Words:",metrics.f1_score(np.array(y_test).reshape(y_pred.shape[0],1), y_pred, average='weighted'))

# The returned "history" object holds a record
# of the loss values and metric values during training
print('\nhistory dict:', history.history)

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss','loss'])


#%% Improve the model - Add tf-idf features
#0.8661030926055311
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(max_features=130)#6000

x_tf_idf = v.fit_transform(data['Processed_Reviews']).toarray()

#x = pad_sequences(x_words, maxlen=maxlen)
y = data[label_column]

x = np.concatenate((x_tf_idf,x_one_hot),axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

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

history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#Plot Loss & Val_loss
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['val_loss','loss'])

y_pred = model.predict(x_test)
mask = y_pred > 0.5
y_pred[mask] = 1
y_pred[~mask] = 0

#y_pred = y_pred.reshape(y_pred.shape[0])
print("Deep Learning Accuracy With One-Hot and tf-idf of Words:",metrics.f1_score(np.array(y_test).reshape(y_pred.shape[0],1), y_pred, average='weighted'))

# The returned "history" object holds a record
# of the loss values and metric values during training
print('\nhistory dict:', history.history)

