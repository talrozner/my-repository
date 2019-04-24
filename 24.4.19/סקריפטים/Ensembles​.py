from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import random
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
 
def put_anomaly(x,y,i_max = 5):
    for i in range(i_max):
        random_index = random.randint(0,149)
        random_column = random.randint(0,3)
        x[random_index,random_column] = x[random_index,random_column] + random.randint(1,5)
        y[random_index] = 3
    return x,y

#Ensembles​
#Bagging - Bootstrap Aggregating ​    
x, y = load_iris(return_X_y=True)
x_new , y_new = put_anomaly(x,y,i_max=50)
x = x_new
y = y_new


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

clf_rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

bagging = BaggingClassifier(clf_rf,max_samples=0.5, max_features=0.5)

clf_rf.fit(X_train,y_train).score(X_test,y_test)
bagging.fit(X_train,y_train).score(X_test,y_test)

#Boosting - AdaBoost
clf_boost = AdaBoostClassifier(clf_rf,n_estimators=100)
clf_boost.fit(X_train,y_train).score(X_test,y_test)

#Blending
model1 = RandomForestClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

clf_voting = VotingClassifier(estimators=[
        ('RandomForestClassifier', model1), ('KNeighborsClassifier', model2), ('LogisticRegression', model3)], voting='hard')#'soft'
    

clf_voting.fit(X_train,y_train).score(X_test,y_test)





















