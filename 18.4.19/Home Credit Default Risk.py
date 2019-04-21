#%%Home Credit Default Risk
#Home Credit Default Risk
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score

def my_f1_score(y_test, y_pred):
    f1_score_ = f1_score(y_test, y_pred, average='weighted')#[0]
    return f1_score_
p_num=1

    
#load data
file_path_0 = r'D:\DS\Seminars\ML Pipeline\Home Credit Default Risk\csv\data\sample_tr_0.csv'
file_path_1 = r'D:\DS\Seminars\ML Pipeline\Home Credit Default Risk\csv\data\sample_tr_1.csv'

data = pd.read_csv(file_path_0)

data = data.append(pd.read_csv(file_path_1))
data.reset_index(inplace = True)
data.drop('index',axis = 1,inplace = True)
#changing data
import random
def put_anomaly(data,label,i_max = 5):
    for i in range(i_max):
        random_index = random.randint(0,199)
        data.loc[random_index,label] = 2
    return data

data = put_anomaly(data,'TARGET',i_max = 5)

#put defect manualy
data.loc[5,'CNT_CHILDREN'] = 100
data.loc[15,'FLAG_OWN_REALTY']=2
data.loc[65,'AMT_CREDIT'] = -1*data.loc[65,'AMT_CREDIT']
data.loc[34,'AMT_CREDIT'] = -1*data.loc[34,'AMT_CREDIT']
data.loc[8,'AMT_CREDIT'] = -1*data.loc[8,'AMT_CREDIT']
data.loc[185,'AMT_GOODS_PRICE'] = -1*data.loc[185,'AMT_GOODS_PRICE']
data.loc[24,'AMT_GOODS_PRICE'] = -1*data.loc[24,'AMT_GOODS_PRICE']

#Type of Features
float64_columns = list(data.dtypes[data.dtypes=='float64'].index)
int64_columns = list(data.dtypes[data.dtypes=='int64'].index)
object_columns = list(data.dtypes[data.dtypes=='object'].index)

#Continuous values Vs Discrete values
continuous_values = list()
discrete_values = list()
for i in data.columns:
    if len(data[i].unique()) > 20:
        continuous_values.append(i)    
    else:
        discrete_values.append(i)
        
new_data = data.iloc[:,:2]
new_data[object_columns[:12]]= data[object_columns[:12]]
new_data[discrete_values[:12]] = data[discrete_values[:12]]
new_data[continuous_values[:12]] = data[continuous_values[:12]]
data = new_data


#################################################################################
#Data - Info
data.info()
data.dtypes

#Type of Features
float64_columns = list(data.dtypes[data.dtypes=='float64'].index)
int64_columns = list(data.dtypes[data.dtypes=='int64'].index)
object_columns = list(data.dtypes[data.dtypes=='object'].index)

#filling na
isna_label = data['TARGET'].isna().any()
columns_na = data.isna().any()

data.fillna(0,inplace = True)


#Data Visualization

#Replace Object to int
data[object_columns] = data[object_columns].astype('category')
data[object_columns] = data[object_columns].apply(lambda x: x.cat.codes)

#Continuous values Vs Discrete values
continuous_values = list()
discrete_values = list()
for i in data.columns:
    if len(data[i].unique()) > 20:
        continuous_values.append(i)    
    else:
        discrete_values.append(i)
        
#finding anomaly
#pie chart to labesl
import matplotlib.pyplot as plt
label_frequncy = data['TARGET'].value_counts()
plt.figure(p_num)
p_num +=1
plt.suptitle("pie chart of category", fontsize=30)
plt.pie(label_frequncy,labels=list(label_frequncy.index), autopct='%1.1f%%', textprops={'fontsize': 30})
plt.show()

#Conclusion - pie chart to labesl 
#There a class that not need to be

#bar plot all features
discrete_frequncy = dict()

for i in discrete_values:  
    discrete_frequncy[i] = data[i].round().astype(int).value_counts()
    
stop = 1
for k,v in discrete_frequncy.items():
    stop += 1
    plt.figure(p_num)
    discrete_frequncy[k].plot(kind='bar', stacked=True)
    plt.suptitle("Bar plot of all categories", fontsize=40)
    plt.xlabel("Value",fontdict={'fontsize': 40})
    plt.ylabel("Frequency",fontdict={'fontsize': 40})
    plt.legend(prop={'size': 30})
    plt.show()
    p_num +=1
    #if stop == 2:
     #   break

#Conclusion - bar plot all features
    #'Target' is anomaly
    #'FLAG_MOBIL' have only 1 value - data['FLAG_MOBIL'].value_counts()
    #The rest of the feature have different distribution
    #'CNT_CHILDREN' cant be 100 (number of children)
    # 'FLAG_OWN_REALTY' cant be 2 (indication of having house)

#'FLAG_MOBIL' - will be drop later

#'CNT_CHILDREN'
mask = data['CNT_CHILDREN']==100
data.loc[mask,'CNT_CHILDREN'] = 1
#'FLAG_OWN_REALTY'
mask = data['FLAG_OWN_REALTY']== 2
data.loc[mask,'FLAG_OWN_REALTY'] = 1

#box plot
import seaborn as sns

plt.figure(p_num)
p_num +=1
sns.boxplot(data = data[continuous_values])
sns.swarmplot(data = data[continuous_values],color=".25")
plt.xlabel("Feature",fontdict={'fontsize': 40})
plt.ylabel("Value",fontdict={'fontsize': 40})

#Two feature have minus values which is not resonable - 'AMT_CREDIT' & 
#'AMT_CREDIT'
mask = data['AMT_CREDIT']<0
data.loc[mask,'AMT_CREDIT'] = -1*data.loc[mask,'AMT_CREDIT']
#'AMT_GOODS_PRICE'
mask = data['AMT_GOODS_PRICE']<0
data.loc[mask,'AMT_GOODS_PRICE'] = -1*data.loc[mask,'AMT_GOODS_PRICE']

#histogram - stacked
continuous_list = list()
continuous_columns_list = list()
for i in continuous_values:
    continuous_list.append(data[i])
    continuous_columns_list.append(i)
    
plt.figure(p_num)
p_num +=1
plt.hist(continuous_list,stacked=True, histtype='bar')
plt.xlabel("Feature",fontdict={'fontsize': 40})
plt.ylabel("Value",fontdict={'fontsize': 40})
plt.legend(continuous_columns_list,prop={'size': 30})

#histogram
for i in continuous_values:
    plt.figure(p_num)
    p_num +=1
    plt.hist(data[i],stacked=True, histtype='bar')
    plt.xlabel("Feature",fontdict={'fontsize': 40})
    plt.ylabel("Value",fontdict={'fontsize': 40})
    plt.legend([i],prop={'size': 30})


#Finding Anomaly by statistical methods
#Gaussian Mixture Models
from sklearn.mixture import GaussianMixture
gau = GaussianMixture(n_components=1)
features = data#data.iloc[:,:56]
gau_score = gau.fit(features)
gau_score = gau.score_samples(features)
gau_score = pd.DataFrame(gau_score)
gau_score.plot(linewidth=7)
plt.suptitle("Gaussian Mixture Models", fontsize=40)
plt.xlabel("Sample",fontdict={'fontsize': 40})
plt.ylabel("Weighted Log Probabilities",fontdict={'fontsize': 40})
p_num+=1
#Sample 8 , 34 and 83 are anomaly - Consider deleting

#PCA
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=3)
features = data.drop('TARGET',axis=1)
labels = data['TARGET']

pca_projecrion = pca.fit_transform(features)

pca_projecrion = pd.DataFrame(pca_projecrion)
fig = plt.figure(p_num, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(pca_projecrion[[0]], pca_projecrion[[1]], pca_projecrion[[2]], c=list(labels.values),cmap=plt.cm.Set1_r, edgecolor='k', s=100)#cmap=plt.cm.Set1
plt.suptitle("PCA", fontsize=40)
plt.show()
p_num+=1
#can't see anomaly!!!

#Hierarchical Clustering​ - dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage  
x = data.drop('TARGET',axis=1) 
linked = linkage(x, 'ward')
max_d = 6e6

plt.figure(p_num,figsize=(10, 7))  
dendrogram(linked,orientation='top'
           ,distance_sort='descending'
           ,show_leaf_counts=True,leaf_font_size = 20,color_threshold = max_d,truncate_mode='lastp',p=150)
plt.axhline(y=max_d, c='k')
plt.suptitle("Hierarchical Clustering", fontsize=40)
plt.xlabel("Sample",fontdict={'fontsize': 40})
plt.ylabel("Dissimilarity",fontdict={'fontsize': 40})
plt.show()  
p_num+=1
#mybe 3 groups are better???

#Hierarchical Clustering​ - AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
cluster.fit_predict(x)  

#AgglomerativeClustering - plot 2D
plt.figure(p_num,figsize=(10, 7))  
plt.scatter(x.loc[:,'AMT_GOODS_PRICE'], x.loc[:,'AMT_CREDIT'],s=100, c=cluster.labels_ , cmap='rainbow') #'rainbow'
plt.suptitle("Agglomerative Clustering", fontsize=40)
plt.xlabel('AMT_GOODS_PRICE',fontdict={'fontsize': 40})
plt.ylabel('AMT_CREDIT',fontdict={'fontsize': 40}) 
plt.show() 
p_num+=1


plt.figure(p_num,figsize=(10, 7))  
plt.scatter(x.loc[:,'AMT_GOODS_PRICE'], x.loc[:,'AMT_CREDIT'],s=100, c=labels , cmap='rainbow') #'rainbow'
plt.suptitle("Agglomerative Clustering", fontsize=40)
plt.xlabel('AMT_GOODS_PRICE',fontdict={'fontsize': 40})
plt.ylabel('AMT_CREDIT',fontdict={'fontsize': 40}) 
plt.show() 
p_num+=1

#AgglomerativeClustering - plot 3D
fig = plt.figure(p_num, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(x.loc[:,'AMT_GOODS_PRICE'], x.loc[:,'AMT_CREDIT'],x.loc[:,'AMT_ANNUITY'], c=labels,cmap=plt.cm.Set1_r, edgecolor='k', s=100)#cmap=plt.cm.Set1
plt.suptitle("AgglomerativeClustering", fontsize=40)
plt.show()
p_num+=1

fig = plt.figure(p_num, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(x.loc[:,'AMT_GOODS_PRICE'], x.loc[:,'AMT_CREDIT'],x.loc[:,'AMT_ANNUITY'], c=cluster.labels_ ,cmap=plt.cm.Set1_r, edgecolor='k', s=100)#cmap=plt.cm.Set1
plt.suptitle("AgglomerativeClustering", fontsize=40)
plt.show()
p_num+=1

#drop 'FLAG_MOBIL'
data.drop('FLAG_MOBIL',axis=1,inplace=True)

#get_dummies
discrete_values.remove('TARGET')
discrete_values.remove('FLAG_MOBIL')
temp_dummies = pd.get_dummies(data[discrete_values].astype('category'))
data.drop(discrete_values,axis=1,inplace=True)
data[temp_dummies.columns] = temp_dummies

#change 'TARGET' == 2 to 0
mask = data['TARGET']==2
data.loc[mask,'TARGET'] = 0

#Train and Test Split
from sklearn.model_selection import train_test_split
x = data.drop('TARGET',axis=1) 
y = data['TARGET']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

################################################################################
#plot learning curve vs sample
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
train_size = list(set(np.round(np.linspace(0.01,0.5,100),2)))
train_size.sort()
train_size = list(train_size)
#train_size = [0.1,0.2,0.3]
clf_rf = RandomForestClassifier(n_estimators = 100,max_depth = None,min_samples_split= 2,max_features = None,verbose = 0)
 
train_sizes, train_scores, test_scores = learning_curve(
    clf_rf, x, y, train_sizes=train_size, cv=2)

plt.figure(p_num)
plt.plot(train_sizes,train_scores[:,0],'ro-',train_sizes,test_scores[:,0],'go-')
plt.legend(['train','test'])
plt.show()
p_num+=1

################################################################################
#without feature selection
df_score_filter_methods = pd.DataFrame(index = ['rf','svc','neigh'])
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = None,min_samples_split= 2,max_features = None,verbose = 0)#,min_samples_leafs = 2

y_pred = clf_rf.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['rf','no fs'] = my_f1_score(y_test, y_pred)


#SVC
from sklearn.svm import SVC
clf_svc = SVC(gamma='auto')
y_pred = clf_svc.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['svc','no fs'] = my_f1_score(y_test, y_pred)

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
clf_neigh = KNeighborsClassifier(n_neighbors=3)

y_pred = clf_neigh.fit(x_train,y_train).predict(x_test)

df_score_filter_methods.loc['neigh','no fs'] = my_f1_score(y_test, y_pred) 




#Feature Selection - Filter Methods
#Chi squared test - GenericUnivariateSelect
from sklearn.feature_selection import GenericUnivariateSelect,chi2
transformer = GenericUnivariateSelect(chi2, 'percentile', param=5)
x = x.abs()#chi test cant get negative values!
x_chi = transformer.fit_transform(x,y)

x_train, x_test, y_train, y_test = train_test_split(x_chi, y, test_size=0.33, random_state=42)

#RandomForestClassifier - with chi2
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = None,min_samples_split= 2,max_features = None,verbose = 0)

y_pred = clf_rf.fit(x_train,y_train).predict(x_test)

df_score_filter_methods.loc['rf','chi2'] = my_f1_score(y_test, y_pred)

#SVC - with chi2
clf_svc = SVC(gamma='auto')
y_pred = clf_svc.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['svc','chi2'] = my_f1_score(y_test, y_pred)

#KNeighborsClassifier - with chi2
clf_neigh = KNeighborsClassifier(n_neighbors=3)
y_pred = clf_neigh.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['neigh','chi2'] = my_f1_score(y_test, y_pred)

#chi2 feature selection improve only random forest
#The results are not stable - some time the unfilter results are better

#%%
#correlation coefficient scores
#correlation between features
x = data.drop('TARGET',axis=1) 
y = data['TARGET']

corr = x.corr()
plt.figure(p_num)
sns.heatmap(corr, annot=True, annot_kws={"size": 40})
plt.suptitle("correlation between features", fontsize=40)
p_num+=1

corr = x.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if np.abs(corr.iloc[i,j]) >= 0.3:
            if columns[j]:
                columns[j] = False
selected_columns = x.columns[columns]
x = x[selected_columns]

#correlation between filtered features
corr = x.corr()
plt.figure(p_num)
sns.heatmap(corr, annot=True, annot_kws={"size": 40})
plt.suptitle("correlation between features", fontsize=40)
p_num+=1

#correlation between label and features
#corr_label = x.corr()['TARGET'].to_frame()
#corr_label.sort_values('TARGET',inplace=True)
#plt.figure(p_num)
#sns.heatmap(corr_label, annot=True, annot_kws={"size": 20})
#plt.suptitle("correlation between label and features", fontsize=40)
#p_num+=1

#split train test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

#RandomForestClassifier - with correlation coefficient scores
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = None,min_samples_split= 2,max_features = None,verbose = 0)

y_pred = clf_rf.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['rf','correlation coefficient scores'] = my_f1_score(y_test, y_pred)


#SVC - with correlation coefficient scores
clf_svc = SVC(gamma='auto')

y_pred = clf_svc.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['svc','correlation coefficient scores'] = my_f1_score(y_test, y_pred)

#KNeighborsClassifier - with correlation coefficient scores
clf_neigh = KNeighborsClassifier(n_neighbors=3)

y_pred = clf_neigh.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['neigh','correlation coefficient scores'] = my_f1_score(y_test, y_pred)
#%%
#correlation coefficient scores improve Random Forest and neigh


#Wrapper Methods
#Recursive Feature Elimination​
from sklearn.feature_selection import RFE
rf_estimator = RandomForestClassifier(n_estimators = 500,max_depth = None,min_samples_split= 2,max_features = None,verbose = 0)
selector = RFE(rf_estimator, 30, step=1)
selector = selector.fit(x, y)
#selector.support_ 
#selector.ranking_
x = x.loc[:,selector.support_ ]

#split train test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

#RandomForestClassifier - with Recursive Feature Elimination​
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = None,min_samples_split= 2,max_features = None,verbose = 0)

y_pred = clf_rf.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['rf','Recursive Feature Elimination'] = my_f1_score(y_test, y_pred)



#SVC - with correlation coefficient scores
clf_svc = SVC(gamma='auto')

y_pred = clf_svc.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['svc','Recursive Feature Elimination'] = my_f1_score(y_test, y_pred)

#KNeighborsClassifier - with correlation coefficient scores
clf_neigh = KNeighborsClassifier(n_neighbors=3)

y_pred = clf_neigh.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['neigh','Recursive Feature Elimination'] = my_f1_score(y_test, y_pred)
#correlation coefficient scores improve Random Forest and neigh
#%%


#Embedded Methods
x = data.drop('TARGET',axis=1) 
y = data['TARGET']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

#RandomForestClassifier - with Embedded Methods​
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = 5,min_samples_split= 10,max_features = 10,verbose = 0)

y_pred = clf_rf.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['rf','Embedded Methods'] = my_f1_score(y_test, y_pred)


#plot learning curve vs iteration
score_clf_rf_train = list()
score_clf_rf_test = list()
for i in range(1,x.shape[1]):
    clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = i,min_samples_split= 10,max_features = 10)
    score_clf_rf_train.append(clf_rf.fit(x_train,y_train).score(x_train,y_train))
    score_clf_rf_test.append(clf_rf.fit(x_train,y_train).score(x_test,y_test))
    print(i)
    print(score_clf_rf_train[-1:])
    print(score_clf_rf_test[-1:])
    
plt.figure(p_num)
plt.plot(score_clf_rf_train,'bo-',score_clf_rf_test,'ro-')
plt.suptitle("RF Score", fontsize=40)
plt.xlabel('max_depth',fontdict={'fontsize': 40})
plt.ylabel('Score',fontdict={'fontsize': 40}) 
plt.legend(['train score','test score'],prop={'size': 35})
plt.show() 
p_num+=1

print("mean score of RF learning curve")
print(np.mean(score_clf_rf_test))

#SVC - with Embedded Methods
clf_svc = SVC(gamma='auto',C=1,probability=True)

y_pred = clf_svc.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['svc','Embedded Methods'] = my_f1_score(y_test, y_pred)

#KNeighborsClassifier - with Embedded Methods
clf_neigh = KNeighborsClassifier(n_neighbors=10)

y_pred = clf_neigh.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['neigh','Embedded Methods'] = my_f1_score(y_test, y_pred)
#KNeighborsClassifier Not learn even the traning!!!
#%%
####################################################################
#for this part i will continue only with random forest
#K-Fold Cross Validation​
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

x = data.drop('TARGET',axis=1) 
y = data['TARGET']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

#RandomForestClassifier - with Embedded Methods​ and K-Fold Cross Validation​
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = 10,min_samples_split= 10,max_features = 10,verbose = 0)

clf_rf_cv = cross_validate(clf_rf, x_train, y_train, cv=30, scoring='f1_weighted',return_estimator=True)

clf_rf_cv['test_score']
clf_rf_cv['test_score'][2]#2 index give the highest score

y_pred = cross_val_predict(clf_rf, x_test, y_test, cv=2)
df_score_filter_methods.loc['rf','Embedded Methods and K-Fold Cross Validation​'] = f1_score(y_test, y_pred, average=None)[0]

#SVC - with Embedded Methods and K-Fold Cross Validation​
clf_svc_cv = cross_validate(clf_svc, x_train, y_train, cv=30, scoring='f1_weighted',return_estimator=True)

clf_svc_cv['test_score']
clf_svc_cv['test_score'][6]#2 index give the highest score

y_pred = cross_val_predict(clf_svc, x_test, y_test, cv=6)
df_score_filter_methods.loc['svc','Embedded Methods and K-Fold Cross Validation​'] = f1_score(y_test, y_pred, average=None)[0]

#KNeighborsClassifier - with Embedded Methods
clf_neigh_cv = cross_validate(clf_neigh, x_train, y_train, cv=30, scoring='f1_weighted',return_estimator=True)

clf_neigh_cv['test_score']
clf_neigh_cv['test_score'][4]#2 index give the highest score

y_pred = cross_val_predict(clf_neigh, x_test, y_test, cv=4)
df_score_filter_methods.loc['neigh','Embedded Methods and K-Fold Cross Validation​'] = f1_score(y_test, y_pred, average=None)[0]
#important to try cv high enough
#%%
#GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[500],'max_depth':[2,5,10],'min_samples_split': [2,5,10,20],'max_features':[5,10,30]}
clf = GridSearchCV(clf_rf, parameters, cv=2)
clf.fit(x_train, y_train)
sorted(clf.cv_results_.keys())

clf.best_estimator_                            
clf.best_params_
clf.best_score_ 
          
df_score_filter_methods.loc['rf','Embedded Methods,K-Fold Cross and GridSearch​'] = clf.best_score_ 

#%% ROC curve AUC
#ROC curve AUC
from sklearn.metrics import roc_curve, auc
y_score = clf_neigh.predict_proba(x_test)

n_classes = y_score.shape[1]

# Compute ROC curve and ROC area for each class
thresholds = list()
for i in range(n_classes):
    temp_test = np.copy(y_test)
    temp_test[temp_test==i]=9
    temp_test[temp_test!=9]=0
    temp_test[temp_test==9]=1
    
    roc_curve_tuple = roc_curve(temp_test, y_score[:, i])
    fpr = roc_curve_tuple[0]
    tpr = roc_curve_tuple[1]
    thresholds = roc_curve_tuple[2]
    roc_auc = auc(fpr, tpr)
    
    plt.figure(p_num)
    plt.plot(fpr,tpr, linewidth=7)
    plt.suptitle("ROC Curve", fontsize=40)
    plt.xlabel("False Positive Rate",fontdict={'fontsize': 40})
    plt.ylabel("True Positive Rate",fontdict={'fontsize': 40})
    plt.legend(['Class - 0','Class - 1'], prop={'size': 35})
    plt.text(1, 1-0.1*i, 'AUC '+ str(i) +' - '+str(round(roc_auc,2)), fontsize=30)

    print(roc_auc)
    
    for k in range(len(thresholds)):
        print("################################")
        print(k)
        print(fpr[k])
        print(tpr[k])
        print(str(round(thresholds[k],2)))
        plt.text(float(fpr[k]), float(tpr[k]), str(round(thresholds[k],2)), fontsize=20)
    plt.show()
p_num+=1

#class 1 threshold = 0.5
#class 0 threshold = 0.51
y_proba = clf_neigh.predict_proba(x_test)

mask_threshold_0 = y_proba[:,0]>=0.5
y_proba[mask_threshold_0,:]=0

mask_threshold_1 = y_proba[:,1]>=0.5
y_proba[mask_threshold_1,:]=1

y_pred = y_proba[:,0]

df_score_filter_methods.loc['neigh','with ROC Curve​'] = f1_score(y_test, y_pred, average=None)[0]

#%% Bootstrap Aggregating 
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(clf_rf,n_estimators = 100,max_samples=0.7, max_features=0.15,bootstrap_features=True)

y_pred = bagging.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['rf','with Bagging'] = my_f1_score(y_test, y_pred)

#clf_rf_cv['estimator'][0]
#%% Boosting
#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf_boost = AdaBoostClassifier(clf_rf,n_estimators=500)#

y_pred = clf_boost.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['rf','AdaBoostClassifier'] = my_f1_score(y_test, y_pred)

#%%Blending
from sklearn.ensemble import VotingClassifier

clf_voting = VotingClassifier(estimators=[
        ('clf_boost', clf_boost), ('clf_rf', clf_rf),('clf_neigh', clf_neigh),('clf_svc', clf_svc)], voting='hard')#'soft'
    
y_pred = clf_voting.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['rf','Blending'] = my_f1_score(y_test, y_pred)


#%% close all plot
plt.close("all")