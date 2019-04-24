from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# split train test
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Confusion Matrix
clf = RandomForestClassifier()
clf_rf = clf.fit(x_train, y_train)
y_pred = clf_rf.predict(x_test)
confusion_matrix(y_test, y_pred)
                    






















