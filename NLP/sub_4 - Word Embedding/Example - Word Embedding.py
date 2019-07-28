import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
p_num=1
file_path = r"D:\DS\NLP\Seminar\Code Example\Targil2\Text parsing and Exploratory Data Analysis.csv"
data = pd.read_csv(file_path)
data = data.iloc[:100]


#%% Word To Vec
#Input to the gensim's Word2Vec can be a list of sentences or list of words or list of list of sentences.
from gensim.models import Word2Vec
sentences = list(data['Processed_Reviews'].str.split(' '))#'Processed_Reviews'
#'Clean_Reviews'

#voc_vec = Word2Vec(sentences, min_count=1)
model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=4)
#model.build_vocab(sentences, update=False)
vectors = model.wv

vocab_list = list(vectors.vocab.keys())

'good' in vocab_list

vectors['good']

vectors.most_similar('good')

print(vectors.similarity('drink', 'alcohol'))
print(vectors.similarity('drink', 'sixteen'))

def get_related_terms(token, topn=10):
    
    for word, similarity in model.most_similar(positive=[token], topn=topn):
        print (word, round(similarity, 3))


get_related_terms('drink', topn=10)

answers = model.most_similar(positive=['tolerance','good','england'], negative=['drink'], topn=3)
print(answers)


#%%Plot Word Vectors Using PCA
from sklearn.decomposition import PCA

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.figure(p_num)
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
p_num+=1

#%% cluster
X = model[model.wv.vocab]

from nltk.cluster import KMeansClusterer
import nltk
num_clusters=10
kclusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)

cluster_dataframe = pd.DataFrame({'cluster':assigned_clusters},index = vocab_list)
cluster_dataframe.sort_values('cluster',inplace=True)

mask = cluster_dataframe['cluster']==0

cluster_dataframe[mask]



#%% Create word DataFrame
'''
# build a list of the terms, integer indices,
# and term counts from the food2vec model vocabulary
ordered_vocab = [(term, voc.index, voc.count) for term, voc in model.wv.vocab.items()]

# sort by the term counts, so the most common terms appear first
ordered_vocab = sorted(ordered_vocab, key=lambda k: -k[2])

# unzip the terms, integer indices, and counts into separate lists
ordered_terms, term_indices, term_counts = zip(*ordered_vocab)
# print(ordered_terms)
# create a DataFrame with the food2vec vectors as data,
# and the terms as row labels
word_vectors = pd.DataFrame(model.wv.syn0norm[term_indices, :], index=ordered_terms)

word_vectors
'''












































