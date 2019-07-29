#%% import
import numpy as np
import pandas as pd

file_path = r"D:\DS\NLP\Seminar\Code Example\Targil2\Text parsing and Exploratory Data Analysis.csv"
data = pd.read_csv(file_path)
data = data.iloc[:1000]

#%%tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=6000)

x_words = vectorizer.fit_transform(data['Processed_Reviews']).toarray()

print(vectorizer.get_feature_names())














