import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#Recursive feature elimination
X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)
selector.support_ 
selector.ranking_


#Recursive feature elimination and cross-validated
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
X, y = make_friedman1(n_samples=50, n_features=100, random_state=0)
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
selector.support_ 
selector.ranking_

# Plot number of features VS. cross-validation scores
plt.figure()
plt.suptitle("Plot number of features VS. cross-validation scores", fontsize=40)
plt.xlabel("Number of features selected",fontdict={'fontsize': 40})
plt.ylabel("Cross validation score",fontdict={'fontsize': 40})
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_,r'-o', linewidth=7.0)
plt.show()

print("Optimal number of features : %d" % selector.n_features_)


#"(nb of correct classifications)"
































##Recursive feature elimination with cross-validation
#import matplotlib.pyplot as plt
#from sklearn.svm import SVC
#from sklearn.model_selection import StratifiedKFold
#from sklearn.feature_selection import RFECV
#from sklearn.datasets import make_classification
#
## Build a classification task using 3 informative features
#X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                           n_redundant=2, n_repeated=0, n_classes=8,
#                           n_clusters_per_class=1, random_state=0)
#
## Create the RFE object and compute a cross-validated score.
#svc = SVC(kernel="linear")
## The "accuracy" scoring is proportional to the number of correct
## classifications
#rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
#              scoring='accuracy')
#rfecv.fit(X, y)
#
#print("Optimal number of features : %d" % rfecv.n_features_)
#
## Plot number of features VS. cross-validation scores
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#plt.show()





















