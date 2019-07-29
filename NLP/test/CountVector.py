#%% import
import numpy as np
import pandas as pd

file_path = r"D:\DS\NLP\Seminar\Code Example\Targil2\Text parsing and Exploratory Data Analysis.csv"
data = pd.read_csv(file_path)
data = data.iloc[:1000]


#%% CountVectorizer -- on Processed Data
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

#get the feature
print(vectorizer.get_feature_names())

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

columns_name = list(vectorizer.get_feature_names())

train_data_features_dataframe = pd.DataFrame(data = train_data_features,columns = columns_name)


















