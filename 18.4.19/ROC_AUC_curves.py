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

def put_anomaly(x,y,i_max = 5):
    for i in range(i_max):
        random_index = random.randint(0,149)
        random_column = random.randint(0,3)
        x[random_index,random_column] = x[random_index,random_column] + random.randint(1,5)
        y[random_index] = 3
    return x,y

# ROC - AUC Curves
x, y = load_iris(return_X_y=True)
x_new , y_new = put_anomaly(x,y,i_max=50)
x = x_new
y = y_new

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0).fit(X_train,y_train)
y_score = clf.predict_proba(X_test)

n_classes = y_score.shape[1]

# Compute ROC curve and ROC area for each class
for i in range(n_classes):
    temp_test = np.copy(y_test)
    temp_test[temp_test==i]=9
    temp_test[temp_test!=9]=0
    temp_test[temp_test==9]=1
    
    roc_curve_tuple = roc_curve(temp_test, y_score[:, i])
    fpr = roc_curve_tuple[0]
    tpr = roc_curve_tuple[1]
    roc_auc = auc(fpr, tpr)
    
    plt.figure(1)
    plt.plot(fpr,tpr, linewidth=7)
    plt.suptitle("ROC Curve", fontsize=40)
    plt.xlabel("False Positive Rate",fontdict={'fontsize': 40})
    plt.ylabel("True Positive Rate",fontdict={'fontsize': 40})
    plt.legend(['Class - 0','Class - 1','Class - 2','Class - 3'], prop={'size': 35})
    plt.text(1, 1-0.1*i, 'AUC '+ str(i) +' - '+str(round(roc_auc,2)), fontsize=30)
    print(roc_auc)


selected_thresh = 0.25245606











