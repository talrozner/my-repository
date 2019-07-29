#%% import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
p_num=1

#%%load data
file_path = r"D:\DS\NLP\Seminar\Code Example\Targil1\Bag_of_Words_Lexicon_Normalization_and_Object_Standardization.csv"

data = pd.read_csv(file_path)

#file_path = r"D:\DS\NLP\toxic-comments-classification\data\train.csv"
#data = pd.read_csv(file_path)
#data.columns = ['id', 'review', 'toxic', 'sentiment', 'obscene', 'threat','insult', 'identity_hate']
#data = data.iloc[:1000]
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
    text = re.sub(r"[,>./<)(]|['']",'',text)
    text = text.lower()
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


#%% Explor Word Frequency
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



