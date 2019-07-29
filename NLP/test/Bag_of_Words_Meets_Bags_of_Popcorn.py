#%% import
import pandas as pd
import numpy as np


#%%load data
file_path = r"D:\DS\NLP\Bag of Words Meets Bags of Popcorn\word2vec-nlp-tutorial\labeledTrainData.tsv"
data = pd.read_table(file_path)

#%%drop column
data.drop('id',inplace=True,axis=1)

data = data[['review','sentiment']]

#%%clean data
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = re.sub(r"[.]",'',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

data['Processed_Reviews'] = data.review.apply(lambda x: clean_text(x))

#clean data - print results
data.head()
data['Processed_Reviews'].apply(lambda x: len(x.split(" "))).mean()

#%% word embeeding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(data['Processed_Reviews'])

maxlen = 130 #according to mean values of words
x = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = data['sentiment']

reverse_word_map = dict(map(reversed,tokenizer.word_index.items()))

#%%train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#%%gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,
    max_depth=10, random_state=0).fit(x_train, y_train)

print('train score ' + str(clf.score(x_train, y_train)))
print('test score ' + str(clf.score(x_test, y_test)))


#%% add dependency tree
'''
import spacy
nlp = spacy.load("en")

def create_dependency_tree(temp):
    doc = nlp(temp)
    dependency_tree = list()
    for token in doc:
        dependency_tree.append([token.lemma_, token.pos_, token.dep_])
    return dependency_tree


data['dependency_trees'] = data.review.apply(lambda x: create_dependency_tree(x))
'''
#%% add Part of speech tagging
'''
from nltk import word_tokenize, pos_tag
def part_of_speech(text):
    tokens = word_tokenize(text)
    return pos_tag(tokens)

data['part_of_speech'] = data.review.apply(lambda x: part_of_speech(x))
'''

#%% NN - train
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

embed_size = 128 #according to mean values of words
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
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#%% NN - test
prediction = model.predict(x_test)
y_pred = (prediction > 0.5)
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)

#%% NN - train F1_Score
from keras import backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

embed_size = 128 #according to mean values of words
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])

batch_size = 100
epochs = 3
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#%% NN - test
prediction = model.predict(x_test)
y_pred = (prediction > 0.5)

print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)






