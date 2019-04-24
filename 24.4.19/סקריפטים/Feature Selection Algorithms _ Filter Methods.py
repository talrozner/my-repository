import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_iris

from sklearn.feature_selection import SelectKBest,SelectPercentile, SelectFpr,SelectFdr,GenericUnivariateSelect,chi2,f_regression

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

#Feature Selection - Filter Methods - Univariate feature selection
# Chi squared test - SelectKBest
x_chi = SelectKBest(chi2, k=2).fit_transform(x, y)

# Chi squared test - SelectPercentile
x_chi = SelectPercentile(chi2, percentile=2).fit_transform(x, y)

# Chi squared test - SelectFpr
x_chi = SelectFpr(chi2, alpha=0.001).fit_transform(x, y)

# Chi squared test - SelectFdr
x_chi = SelectFdr(chi2, alpha=0.001).fit_transform(x, y)

# Chi squared test - GenericUnivariateSelect - mode : {‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’}
transformer = GenericUnivariateSelect(chi2, 'k_best', param=2)
x_chi = transformer.fit_transform(x, y)

#These objects take as input a scoring function that returns univariate scores and p-values (or only scores for SelectKBest and SelectPercentile):

#For regression: f_regression, mutual_info_regression
#For classification: chi2, f_classif, mutual_info_classif

#Chi squared test - GenericUnivariateSelect - for regression
transformer = GenericUnivariateSelect(chi2, 'k_best', param=2)
transformer.score_func(x,y)
x_chi = transformer.fit_transform(x, y)





















