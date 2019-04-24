from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
import time
import numpy as np
import pandas as pd
import seaborn as sns

p_num = 1

def put_anomaly(x,y,i_max = 5):
    for i in range(i_max):
        random_index = random.randint(0,149)
        random_column = random.randint(0,3)
        x[random_index,random_column] = x[random_index,random_column] + random.randint(1,5)
        y[random_index] = 3
    return x,y
    
    
x, y = load_iris(return_X_y=True)
x_new , y_new = put_anomaly(x,y)

data = pd.DataFrame(data = x_new)
data['label'] = y_new

#pie chart
#pie chart to labesl
label_frequncy = data['label'].value_counts()
plt.figure(p_num)
p_num +=1
plt.suptitle("pie chart of category", fontsize=30)
plt.pie(label_frequncy,labels=list(label_frequncy.index), autopct='%1.1f%%', textprops={'fontsize': 30})
plt.show()

#pie chart to feature
feature_frequncy_0 = data[0].round().astype(int).value_counts()
plt.figure(p_num)
p_num +=1
plt.suptitle("pie chart of category 0", fontsize=30)
plt.pie(feature_frequncy_0,labels=list(feature_frequncy_0.index), autopct='%1.1f%%', textprops={'fontsize': 30})
plt.show()

#bar plot - feature 0
plt.figure(p_num)
p_num +=1
plt.suptitle("Bar plot of category 0", fontsize=40)
feature_frequncy_0.plot(kind='bar')
plt.xlabel("Value",fontdict={'fontsize': 40})
plt.ylabel("Frequency",fontdict={'fontsize': 40})
plt.show()

#bar plot - all feature 
feature_frequncy_0 = data[0].round().astype(int).value_counts()
feature_frequncy_1 = data[1].round().astype(int).value_counts()
feature_frequncy_2 = data[2].round().astype(int).value_counts()
feature_frequncy_3 = data[3].round().astype(int).value_counts()

index = [x for x in range(int(data.min().min()),int(data.max().max())+1)]

feature_frequncy = pd.DataFrame(index = index)
feature_frequncy.loc[list(feature_frequncy_0.index),'feature 0'] = feature_frequncy_0
feature_frequncy.loc[list(feature_frequncy_1.index),'feature 1'] = feature_frequncy_1
feature_frequncy.loc[list(feature_frequncy_2.index),'feature 2'] = feature_frequncy_2
feature_frequncy.loc[list(feature_frequncy_3.index),'feature 3'] = feature_frequncy_3



plt.figure(p_num)
feature_frequncy.plot(kind='bar', stacked=True)
plt.suptitle("Bar plot of all categories", fontsize=40)
plt.xlabel("Value",fontdict={'fontsize': 40})
plt.ylabel("Frequency",fontdict={'fontsize': 40})
plt.legend(prop={'size': 30})
plt.show()
p_num +=1

#box plot
plt.figure(p_num)
p_num +=1
sns.boxplot(data = data.iloc[:,:4])
sns.swarmplot(data = data.iloc[:,:4],color=".25")
plt.xlabel("Feature",fontdict={'fontsize': 40})
plt.ylabel("Value",fontdict={'fontsize': 40})

#histogram
plt.figure(p_num)
p_num +=1
plt.hist([data[0],data[1],data[2],data[3]],stacked=True, histtype='bar')
plt.xlabel("Feature",fontdict={'fontsize': 40})
plt.ylabel("Value",fontdict={'fontsize': 40})
plt.legend(data.columns[:4],prop={'size': 30})








