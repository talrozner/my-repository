import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
file_path = r"D:\DS\NLP\Seminar\Code Example\Code By Subject\sub_3 - Feature Engineering\Text parsing and Exploratory Data Analysis.csv"
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
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    return x,y,reverse_word_map

max_features = 100000#6000
maxlen = 130#130

x,y, reverse_word_map = feature_engineering(data,'Processed_Reviews','sentiment',max_features,maxlen)

#Feature Selection Using chi2 improve the results
from sklearn.feature_selection import GenericUnivariateSelect, chi2
#Chi squared test - GenericUnivariateSelect - for regression
transformer = GenericUnivariateSelect(chi2, 'percentile', param=10)
transformer.score_func(x,y)
x_chi = transformer.fit_transform(x, y)

x = x_chi


# Split x and y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Model Generation Using RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0).fit(x_train, y_train)
predicted= clf.predict(x_test)

print("RandomForestClassifier F1 Score With One-Hot Encoding: ",metrics.f1_score(y_test, predicted))
print("\n")

#%%tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(max_features=6000)#6000-80%,130-72%,20000-80%

x = v.fit_transform(data['Processed_Reviews']).toarray()

y = data['sentiment']

#Feature Selection Using chi2 not improve the results
'''
from sklearn.feature_selection import GenericUnivariateSelect, chi2
#Chi squared test - GenericUnivariateSelect - for regression
transformer = GenericUnivariateSelect(chi2, 'percentile', param=10)
transformer.score_func(x,y)
x_chi = transformer.fit_transform(x, y)

x = x_chi
'''

# Split x and y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Model Generation Using RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0).fit(x_train, y_train)
predicted= clf.predict(x_test)

print("RandomForestClassifier F1 Score With tf-idf: ",metrics.f1_score(y_test, predicted))
print("\n")



#%% CountVectorizer - train on chooes num of words
from sklearn.feature_extraction.text import CountVectorizer

max_features_list = list()
f1_score_list = list()

index_validation = list(data['Processed_Reviews'].sample(frac=0.3).index)
data_train = data['Processed_Reviews'].iloc[~data['Processed_Reviews'].isin(index_validation).values]

label_train = data["sentiment"].iloc[~data["sentiment"].isin(index_validation).values]

for i in range(100,1000,100):
    print(i)
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = i) #6000,130
    
    # The input to fit_transform should be a list of strings.
    x = vectorizer.fit_transform(data_train)
    
    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    x = x.toarray()
    x = pad_sequences(x, maxlen=maxlen)
    
    y = label_train
    
    #splitting dataset into training and testing data
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    
    # Model Generation Using RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0).fit(x_train, y_train)
    predicted= clf.predict(x_test)
    
    f1_score_list.append(metrics.f1_score(y_test, predicted))
    max_features_list.append(i)
    #print("RandomForestClassifier F1 Score With CountVectorizer: ",metrics.f1_score(y_test, predicted))
    #print("\n")

plt.plot(max_features_list,f1_score_list)

#%% CountVectorizer - validation on best num of words (800)

data_train = data['Processed_Reviews'].iloc[index_validation]

label_train = data["sentiment"].iloc[index_validation]


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 800)

# The input to fit_transform should be a list of strings.
x = vectorizer.fit_transform(data_train)

# Numpy arrays are easy to work with, so convert the result to an 
# array
x = x.toarray()
x = pad_sequences(x, maxlen=maxlen)

y = label_train

#splitting dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

# Model Generation Using RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0).fit(x_train, y_train)
predicted= clf.predict(x_test)

print("RandomForestClassifier F1 Score With CountVectorizer of 800 words: ",metrics.f1_score(y_test, predicted))
print("\n")

























