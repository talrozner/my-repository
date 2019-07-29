import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import requests
import re
from collections import Counter 
def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))

#ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')

p_num=1

list_file_path = [r"D:\DS\NLP\Seminar\Code Example\Targil2\Data Of Articles\articles1.csv",r"D:\DS\NLP\Seminar\Code Example\Targil2\Data Of Articles\articles2.csv",r"D:\DS\NLP\Seminar\Code Example\Targil2\Data Of Articles\articles3.csv"]

data = pd.DataFrame()
for i in list_file_path:
    temp_data = pd.read_csv(i)
    if data.shape[0] == 0:
        data = temp_data
    else:
        data = data.append(temp_data)
    

mask_url_data = data['url'].notnull()

url_data = data.loc[mask_url_data,'url']
url_data = url_data.iloc[:20]
news_list = list()
for url in url_data:
    text = url_to_string(url)
    news_list.append(sent_tokenize(text))


#%% Word To Vec
#Input to the gensim's Word2Vec can be a list of sentences or list of words or list of list of sentences.
from gensim.models import Word2Vec
sentences = news_list
model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=4)

vectors = model.wv
#%%Plot Word Vectors Using PCA
from sklearn.decomposition import PCA

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.figure(p_num)
plt.scatter(result[:, 0], result[:, 1])
#words = list(model.wv.vocab)
words = [i for i in range(0,len(sentences))]
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
p_num+=1

#%% cluster
vocab_list = words
X = model[model.wv.vocab]

from nltk.cluster import KMeansClusterer
import nltk
num_clusters=2
kclusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)

cluster_dataframe = pd.DataFrame({'cluster':assigned_clusters},index = vocab_list)

mask = cluster_dataframe['cluster']==0

cluster_dataframe[mask]
















