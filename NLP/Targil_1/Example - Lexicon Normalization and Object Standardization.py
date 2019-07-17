#%%import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%--Text Preprocessing -- Noise Removal

import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 

def clean_text(text):
    text = re.sub(r"[@][a-zA-Z0-9]+|[.]",'',text)
    text = text.lower()
    text = [word for word in text.split(' ') if not word in stop_words]
    text = " ".join(text)
    return text

text = r"@VirginAmerica plus you've added commercials to the experience... tacky."

new_text = clean_text(text)

print("clean text: " +"\n" + str(new_text))


#%%--Text Preprocessing ----Lexicon Normalization -- Stemming
from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word = "multiplying" 

print('\n'+"original word: "+'\n'+word+'\n'+'\n'+"stem word: "+"\n"+  stem.stem(word))

#%%--Text Preprocessing ----Lexicon Normalization -- Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

print('\n'+'lemma word with pos v:' +"\n"+  lem.lemmatize(word, "v"))

#%%--Text Preprocessing -- Object Standardization
lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love"}

def _lookup_words(input_text):
    words = input_text.split() 
    new_words = [] 
    for word in words:
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
        new_words.append(word)
        new_text = " ".join(new_words) 
    return new_text

_lookup_words("RT this is a retweeted tweet by Shivam Bansal")

#%% Tokenize words and Tokenizing sentences
from nltk.tokenize import sent_tokenize, word_tokenize

data = "All work and no play makes jack a dull boy, all work and no play"
print("Word Tokenize")
print(word_tokenize(data))

print("Sentences Tokenize")
print(sent_tokenize(data))



#######################################################################################
#%%load data
file_path = r"D:\DS\NLP\Seminar\Code Example\Targil1\Bag_of_Words_Lexicon_Normalization_and_Object_Standardization.csv"

data = pd.read_csv(file_path)

#%%drop column
data.drop('id',inplace=True,axis=1)

data = data[['review','sentiment']]

#data = data.iloc[:100]

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
frequency_word = create_frequency_df(data,'sentiment','review',num_of_words)
plot_freq(frequency_word)





















