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
    scaler = StandardScaler()
    X_scaler = scaler.fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.33, random_state=42)
    X_train[0:10,:] = X_train[0:10,:]*i#*random.random()
    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    
    start_traning_time = time.time()
    logreg.fit(X_train, y_train)
    traning_time.append(time.time() - start_traning_time)
    
    logreg.predict(X_test)
    score_list.append(logreg.score(X_test,y_test))
    

my_plot_score(score_list,"Logistic Regression")
my_plot_traning_time(traning_time,"Logistic Regression")



X, y = load_iris(return_X_y=True)

score_list = list()

for i in range(100):
    scaler = StandardScaler()
    X_scaler = scaler.fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.33, random_state=42)
    X_train[0:10,:] = X_train[0:10,:]*i#*random.random()
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    
    start_traning_time = time.time()
    clf.fit(X_train, y_train)
    traning_time.append(time.time() - start_traning_time)
    
    score_list.append(clf.score(X_test,y_test))

my_plot_score(score_list,"Random Forest")
my_plot_traning_time(traning_time,"Random Forest")



X, y = load_iris(return_X_y=True)

score_list = list()

for i in range(100):
    scaler = StandardScaler()
    X_scaler = scaler.fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.33, random_state=42)
    X_train[0:10,:] = X_train[0:10,:]*i#*random.random()
    kmeans = KMeans(n_clusters=3, random_state=0)
    
    start_traning_time = time.time() 
    kmeans.fit(X_train)
    traning_time.append(time.time() - start_traning_time)
    
    kmeans.labels_    
    kmeans.cluster_centers_
    y_pred = kmeans.predict(X_test)
    score_list.append(metrics.adjusted_rand_score(y_test, y_pred))
    #score_list.append(kmeans.score(X_test, y_test))
    
my_plot_score(score_list,"K-Means")
my_plot_traning_time(traning_time,"K-Means")




















