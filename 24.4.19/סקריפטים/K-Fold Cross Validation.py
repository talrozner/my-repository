from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# K-Fold Cross Validation
x, y = load_iris(return_X_y=True)
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, x, y, cv=5, scoring='accuracy')
print("Test Score")
print(scores['test_score'])

print("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))


#GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(x, y)
sorted(clf.cv_results_.keys())

clf.best_estimator_                            
clf.best_params_
clf.best_score_                       






















