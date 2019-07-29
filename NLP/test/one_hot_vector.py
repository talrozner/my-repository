#%% import
import numpy as np
import pandas as pd

file_path = r"D:\DS\NLP\Seminar\Code Example\Targil2\Text parsing and Exploratory Data Analysis.csv"
data = pd.read_csv(file_path)
data = data.iloc[:1000]
#%% Data Exploration - One Hot Vactor
mean_len = data['Processed_Reviews'].apply(lambda x: len(x.split(" "))).mean()
print("The mean number of word in sentance is " + str(mean_len))

maxlen = 130

#uniqe_words = set()
#data['Processed_Reviews'].str.lower().str.split().apply(uniqe_words.update)
#num_of_uniqe_words = len(uniqe_words)

uniqe_words = data['Processed_Reviews'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)

uniqe_words.sort_values(ascending=False,inplace=True)

print(uniqe_words)
print(len(uniqe_words))

max_features = 6000

#%% feature engineering  - One Hot Vactor
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(data['Processed_Reviews'])

#maxlen = 130
x = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = data['sentiment']
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


#%% feature engineering - function - One Hot Vactor
def feature_engineering(data,column,label_column,max_features,maxlen):
    #max_features = 6000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(data[column])#'Processed_Reviews'
    list_tokenized_train = tokenizer.texts_to_sequences(data[column])#'Processed_Reviews'
    
    #maxlen = 130
    x = pad_sequences(list_tokenized_train, maxlen=maxlen)
    y = data[label_column]#'sentiment'
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    return x,y,reverse_word_map











