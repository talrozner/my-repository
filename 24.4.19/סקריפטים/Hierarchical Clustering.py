import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

p_num = 1

x, y = load_iris(return_X_y=True)

data = pd.DataFrame(data = x)
data['label'] = y


#Hierarchical Clustering​ - dendrogram
linked = linkage(x, 'ward')
max_d = 5

plt.figure(p_num,figsize=(10, 7))  
dendrogram(linked,orientation='top'
           ,distance_sort='descending'
           ,show_leaf_counts=True,leaf_font_size = 20,color_threshold = max_d,truncate_mode='lastp',p=150)
plt.axhline(y=max_d, c='k')
plt.suptitle("Hierarchical Clustering", fontsize=40)
plt.xlabel("Sample",fontdict={'fontsize': 40})
plt.ylabel("Dissimilarity",fontdict={'fontsize': 40})
plt.show()  
p_num+=1



#Hierarchical Clustering​ - AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
cluster.fit_predict(x)  

plt.figure(p_num,figsize=(10, 7))  
plt.scatter(x[:,0], x[:,1],s=100, c=cluster.labels_, cmap='rainbow') #'rainbow'
plt.suptitle("Agglomerative Clustering", fontsize=40)
plt.xlabel("X0",fontdict={'fontsize': 40})
plt.ylabel("X1",fontdict={'fontsize': 40}) 
plt.show() 








