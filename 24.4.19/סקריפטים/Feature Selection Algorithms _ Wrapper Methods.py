import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#%%data
x, y = make_friedman1(n_samples=50, n_features=100, random_state=0)


#%%No FS
clf_svr = SVR(kernel="linear")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf_svr_fit = clf_svr.fit(x_train,y_train)

y_pred_train = clf_svr_fit.predict(x_train)
y_pred_test = clf_svr_fit.predict(x_test)

y_rf_train_score = mean_squared_error(y_train, y_pred_train)
y_rf_test_score = mean_squared_error(y_test, y_pred_test)

print("The train score without FS is " + str(y_rf_train_score))
print("The test score without FS is " + str(y_rf_test_score))
#%%Recursive feature elimination
estimator = SVR(kernel="linear")
selector = RFE(estimator,n_features_to_select = 5, step=1)
selector = selector.fit(x_train, y_train)
#selector.support_ 
#selector.ranking_

y_pred_train = selector.predict(x_train)
y_pred_test = selector.predict(x_test)

y_rf_rfe_train_score = mean_squared_error(y_train, y_pred_train)
y_rf_rfe_test_score = mean_squared_error(y_test, y_pred_test)

print("The train score with RFE is " + str(y_rf_rfe_train_score))
print("The test score with RFE is " + str(y_rf_rfe_test_score))

#%%Recursive feature elimination - curve
y_rfe_train_score = list()
y_rfe_test_score = list()
n_features =list()
for i in range(1,x.shape[1]):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator,n_features_to_select = i, step=1)
    selector = selector.fit(x_train, y_train)    
    
    n_features.append(selector.n_features_)
    
    y_pred_train = selector.predict(x_train)
    y_pred_test = selector.predict(x_test)
    
    y_rfe_train_score.append(mean_squared_error(y_train, y_pred_train))
    y_rfe_test_score.append(mean_squared_error(y_test, y_pred_test))
    
    
#%% mse vs feature number
plt.figure()
plt.suptitle("mse vs feature number", fontsize=40)
plt.xlabel("Number of features selected",fontdict={'fontsize': 40})
plt.ylabel("mse score",fontdict={'fontsize': 40})
plt.plot(n_features,y_rfe_train_score,'r',linewidth=7.0)
plt.plot(n_features,y_rfe_test_score,'g',linewidth=7.0)
plt.legend(['train score','test score'],prop={'size': 30})
plt.show()



#"(nb of correct classifications)"
    
#%%Recursive feature elimination CV  
from sklearn.feature_selection import RFECV

y_rfe_train_score = list()
y_rfe_test_score = list()
n_features =list()
for i in range(1,x.shape[1]):
    estimator = SVR(kernel="linear")
    selector = RFECV(estimator,min_features_to_select=i, step=1, cv=5)

    selector = selector.fit(x_train, y_train)    
    
    n_features.append(selector.n_features_)
    
    y_pred_train = selector.predict(x_train)
    y_pred_test = selector.predict(x_test)
    
    y_rfe_train_score.append(mean_squared_error(y_train, y_pred_train))
    y_rfe_test_score.append(mean_squared_error(y_test, y_pred_test))

#%% Recursive feature elimination CV 
#mse vs feature number
plt.figure()
plt.suptitle("mse vs feature number", fontsize=40)
plt.xlabel("Number of features selected",fontdict={'fontsize': 40})
plt.ylabel("mse score",fontdict={'fontsize': 40})
plt.plot(n_features,y_rfe_train_score,'r',linewidth=7.0)
plt.plot(n_features,y_rfe_test_score,'g',linewidth=7.0)
plt.legend(['train score','test score'],prop={'size': 30})
plt.show()
































##Recursive feature elimination with cross-validation
#import matplotlib.pyplot as plt
#from sklearn.svm import SVC
#from sklearn.model_selection import StratifiedKFold
#from sklearn.feature_selection import RFECV
#from sklearn.datasets import make_classification
#
## Build a classification task using 3 informative features
#x, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                           n_redundant=2, n_repeated=0, n_classes=8,
#                           n_clusters_per_class=1, random_state=0)
#
## Create the RFE object and compute a cross-validated score.
#svc = SVC(kernel="linear")
## The "accuracy" scoring is proportional to the number of correct
## classifications
#rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
#              scoring='accuracy')
#rfecv.fit(x, y)
#
#print("Optimal number of features : %d" % rfecv.n_features_)
#
## Plot number of features VS. cross-validation scores
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#plt.show()





















