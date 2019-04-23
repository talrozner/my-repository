import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score

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

data = pd.DataFrame(data = x_new)
data['label'] = y

x = x_new[:,0:2].copy()

#split train test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#Random Forest without - Ebmedded Methods
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf_rf = clf.fit(x_train, y_train)
y_pred = clf_rf.predict(x_test)

print("f1 score Random Forest - " + str(f1_score(y_test, y_pred, average='weighted')))

#Random Forest with Ebmedded Methods 
clf = RandomForestClassifier(n_estimators=500,max_depth=10,min_samples_split=10,min_samples_leaf=10,max_features=0.5)
clf_rf = clf.fit(x_train, y_train)
y_pred = clf_rf.predict(x_test)

print("f1 score Embedded Random Forest - " + str(f1_score(y_test, y_pred, average='weighted')))





#SVC - Ebmedded Methods
from sklearn.svm import SVC

clf = SVC(gamma='auto')
clf_svc = clf.fit(x_train, y_train)
y_pred = clf_svc.predict(x_test)

print("f1 score SVC - " + str(f1_score(y_test, y_pred, average='weighted')))

#SVC with Ebmedded Methods 
clf = SVC(gamma='auto',C=1000)

clf_svc = clf.fit(x_train, y_train)
y_pred = clf_svc.predict(x_test)

print("f1 score Embedded SVC - " + str(f1_score(y_test, y_pred, average='weighted')))




