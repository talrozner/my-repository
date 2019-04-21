import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib
import time
from sklearn.model_selection import train_test_split

start_time = time.time()
p_num = 1

def my_plot_score(score_list,title,fontsize = 30):
    global p_num
    plt.figure(p_num)
    p_num +=1
    plt.plot(score_list,'o-',linewidth=7.0)
    plt.ylabel("Score",fontsize = fontsize)
    plt.xlabel("Added value of anomaly",fontsize = fontsize)
    plt.title(title,fontsize = fontsize)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)    

def my_plot_traning_time(score_list,title,fontsize = 30):
    global p_num
    plt.figure(p_num)
    p_num +=1
    plt.plot(score_list,'o-',linewidth=7.0)
    plt.ylabel("Traning time",fontsize = fontsize)
    plt.xlabel("Added value of anomaly",fontsize = fontsize)
    plt.title(title,fontsize = fontsize)
    


X, y = load_iris(return_X_y=True)

score_list = list()
traning_time = list()

for i in range(100):
    print(i)
    scaler = StandardScaler()
    X_scaler = scaler.fit(X).transform(X)
    X_scaler = np.insert(X_scaler,[0],X_scaler*(i+4),axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.33, random_state=42)
    
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    
    start_traning_time = time.time()
    clf.fit(X_train, y_train)
    traning_time.append(time.time() - start_traning_time)
    
    score_list.append(clf.score(X_test,y_test))

my_plot_score(score_list,"Random Forest")
my_plot_traning_time(traning_time,"Random Forest")









