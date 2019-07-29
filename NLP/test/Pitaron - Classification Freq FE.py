import numpy as np
import pandas as pd

#file_path = r"D:\DS\NLP\Seminar\Code Example\Targil3\DataForTargil3.csv"
file_path = r"D:\DS\NLP\Seminar\Code Example\Targil2\Text parsing and Exploratory Data Analysis.csv"
data = pd.read_csv(file_path)


#data = data[['review', 'sentiment', 'Clean_Reviews','Processed_Reviews', 'tag dependency', 'pos dependency','dep dependency', 'Entities', 'ngrams']]

#data['Entities'] = data['Entities'].str.lower()
#data['Entities'].fillna('Nan',inplace=True)
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
    #reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    return x#,reverse_word_map

#'pos tag','pos dependency', 'dep dependency'
label_column = 'sentiment'
max_features = 6000
maxlen = 130


x_words = feature_engineering(data,'Processed_Reviews',label_column,max_features,maxlen)

#x_tag_dependency = feature_engineering(data,'tag dependency',label_column,max_features,maxlen)

#x_pos_dependency = feature_engineering(data,'pos dependency',label_column,max_features,maxlen)

#x_dep_dependency = feature_engineering(data,'dep dependency',label_column,max_features,maxlen)

#x_entities = feature_engineering(data,'Entities',label_column,max_features,maxlen)

#x_ngrams = feature_engineering(data,'ngrams',label_column,max_features,maxlen)

#x = np.append(x_words,x_tag_dependency,axis=1)
#x = np.append(x,x_pos_dependency,axis=1)
#x = np.append(x,x_dep_dependency,axis=1)
#x = np.append(x,x_entities,axis=1)
#x = np.append(x,x_ngrams,axis=1)

x = x_words
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
print("RandomForestClassifier F1 Score With One-Hot Encoding:",metrics.f1_score(y_test, predicted))

#from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, max_depth=2,random_state=0).fit(x_train, y_train)

#predicted= clf.predict(x_test)
#print("GradientBoostingClassifier F1 Score With One-Hot Encoding:",metrics.f1_score(y_test, predicted))

print("\n")
#%%tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(max_features=6000)

x_words = v.fit_transform(data['Processed_Reviews']).toarray()
#x_tag_dependency = v.fit_transform(data['tag dependency']).toarray()
#x_pos_dependency = v.fit_transform(data['pos dependency']).toarray()
#x_dep_dependency = v.fit_transform(data['dep dependency']).toarray()
#x_entities = v.fit_transform(data['Entities']).toarray()
#x_ngrams = v.fit_transform(data['ngrams']).toarray()

#x = np.append(x_words,x_tag_dependency,axis=1)
#x = np.append(x,x_pos_dependency,axis=1)
#x = np.append(x,x_dep_dependency,axis=1)
#x = np.append(x,x_entities,axis=1)
#x = np.append(x,x_ngrams,axis=1)
x = x_words
y = data[label_column]


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
print("RandomForestClassifier Accuracy With tf-idf:",metrics.f1_score(y_test, predicted))


#from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, max_depth=2,random_state=0).fit(x_train, y_train)
#predicted= clf.predict(x_test)
#print("GradientBoostingClassifier F1 Score With tf-idf:",metrics.f1_score(y_test, predicted))

print("\n")
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

clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0)
clf.fit(x_train, y_train)  

predicted= clf.predict(x_test)
print("RandomForest Accuracy With CountVectorizer of Words:",metrics.f1_score(y_test, predicted))

#from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, max_depth=2,random_state=0).fit(x_train, y_train)
#predicted= clf.predict(x_test)
#print("GradientBoostingClassifier F1 Score With CountVectorizer of Words:",metrics.f1_score(y_test, predicted))

print("\n")






























