from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[0, 0], [1, 1], [3,3],[5, 5]])
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
indices                                           
distances

from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))


from sklearn.neighbors import KernelDensity
kde = KernelDensity(bandwidth=0.04, metric='haversine',
                        kernel='gaussian', algorithm='ball_tree',n)

#kde.fit(Xtrain[ytrain == i])
kde.fit(X)
kde.score_samples(X)

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

X, y = load_iris(return_X_y=True)
km = KMeans(n_clusters = 5, random_state = 1).fit(X)

dists = euclidean_distances(km.cluster_centers_)

import numpy as np
tri_dists = dists[np.triu_indices(5, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()




















