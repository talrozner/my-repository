import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

p_num = 1

x, y = load_iris(return_X_y=True)

data = pd.DataFrame(data = x)
data['label'] = y

#correlation coefficient scores
#Selecting features based on correlation
X = data[[0,1,2,3]]
y = data['label']

#correlation between features
corr = X.corr()
plt.figure(p_num)
sns.heatmap(corr, annot=True, annot_kws={"size": 40})
plt.suptitle("correlation between features", fontsize=40)
p_num+=1

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = X.columns[columns]
X = X[selected_columns]

#correlation between label and features
corr_label = data.corr()['label'].to_frame()
corr_label.sort_values('label',inplace=True)
plt.figure(p_num)
sns.heatmap(corr_label, annot=True, annot_kws={"size": 20})
plt.suptitle("correlation between label and features", fontsize=40)
p_num+=1







c = X.corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort")
so_filter = so[so>0.5]

column_set = set()
for i,j in so_filter.index:
    column_set.add(i)
    column_set.add(j)


print(column_set)














