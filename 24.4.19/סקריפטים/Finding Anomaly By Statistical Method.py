import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import random
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

p_num = 1

def put_anomaly(x,i_max = 5):
    for i in range(i_max):
        random_index = random.randint(0,149)
        random_column = random.randint(0,3)
        x[random_index,random_column] = x[random_index,random_column] + random.randint(1,5)
        x[0,:] = 6
    return x
    
    
x, y = load_iris(return_X_y=True)
x_new  = put_anomaly(x)
#x_new = x

data = pd.DataFrame(data = x_new)
data['label'] = y

#finding anomaly by z-score
data_z_score = pd.DataFrame()
for i in list(data.columns):
    if i == 'label':
        continue
    else:
        data_z_score[i] = (data[i]-data[i].mean())/data[i].std()#ddof=0

data_z_score.plot()
plt.suptitle("Z-Score of all Features", fontsize=40)
plt.xlabel("Sample",fontdict={'fontsize': 40})
plt.ylabel("Z-Score",fontdict={'fontsize': 40})
plt.legend(["Feature - " + str(x) for x in data_z_score.columns],prop={'size': 25})
p_num+=1





#Gaussian Mixture Models
gau = GaussianMixture(n_components=1)
features = data[[0,1,2,3]]
gau_score = gau.fit(features)
gau_score = gau.score_samples(features)
gau_score = pd.DataFrame(gau_score)
gau_score.plot(linewidth=7)
plt.suptitle("Gaussian Mixture Models", fontsize=40)
plt.xlabel("Sample",fontdict={'fontsize': 40})
plt.ylabel("Weighted Log Probabilities",fontdict={'fontsize': 40})
p_num+=1




#PCA
pca = PCA(n_components=3)
features = data[[0,1,2,3]]
labels = data['label']

pca_projecrion = pca.fit_transform(features)

pca_projecrion = pd.DataFrame(pca_projecrion)
fig = plt.figure(p_num, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(pca_projecrion[[0]], pca_projecrion[[1]], pca_projecrion[[2]], c=list(labels.values),cmap=plt.cm.Set1, edgecolor='k', s=100)
plt.suptitle("PCA", fontsize=40)
plt.show()
p_num+=1





#Nearest Neighbors
features = data[[0,1,2,3]]
nbrs = NearestNeighbors(n_neighbors=2).fit(features)
distances, indices = nbrs.kneighbors(features)
distances = pd.DataFrame(distances)
indices = pd.DataFrame(indices)
distances[1].plot(linewidth=7)
plt.suptitle("Distance from Nearest Neighbors", fontsize=40)
plt.xlabel("Sample",fontdict={'fontsize': 40})
plt.ylabel("Distance",fontdict={'fontsize': 40})
p_num+=1 




#KMeans Clustering Distance
km = KMeans(n_clusters = 3, random_state = 1).fit(features)
dists = euclidean_distances(features,km.cluster_centers_)
dists = pd.DataFrame(dists)

dist_r = pd.DataFrame(np.square(dists[0]**2 + dists[1]**2 + dists[2]**2))

dist_r.plot(linewidth=7)
plt.suptitle("KMeans Distance from Cluster Center", fontsize=40)
plt.xlabel("Sample",fontdict={'fontsize': 40})
plt.ylabel("Distance",fontdict={'fontsize': 40})
p_num+=1 


#KMeans Clustering - finding number of clustering
#score_list = list()
#i_max = 100
#for i in range(2,i_max):
#    km = (KMeans(n_clusters = i, random_state = 1).fit(x)).inertia_#.fit(features)
#    score_list.append(km)
#
#plt.figure(p_num)    
#plt.plot([i for i in range(2,i_max)],score_list)
#p_num+=1 



