#Home Credit Default Risk
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

p_num=1

    
#load data
file_path_0 = r'D:\DS\Seminars\ML Pipeline\Home Credit Default Risk\csv\data\sample_tr_0.csv'
file_path_1 = r'D:\DS\Seminars\ML Pipeline\Home Credit Default Risk\csv\data\sample_tr_1.csv'

data = pd.read_csv(file_path_0)

data = data.append(pd.read_csv(file_path_1))
data.reset_index(inplace = True)
data.drop('index',axis = 1,inplace = True)
data.drop('FLAG_MOBIL',axis = 1,inplace = True)
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

#%%Fill na
columns_na = data.columns[data.isna().any()]

for i in continuous_values:
    filling_value = pd.DataFrame(data[i]).mean()
    data[i] = pd.DataFrame(data[i]).fillna(filling_value)
    
for i in discrete_values:
    if i=='TARGET':
        continue
    freq_val = data[i].value_counts()
    max_freq_value = freq_val.max()
    max_value = freq_val[freq_val==max_freq_value].index
    if max_value.values.dtype =='O':
        filling_value = str(max_value.values)
    elif type(max_value.values) is np.ndarray:
        filling_value = float(max_value.values[0])
    else:
        filling_value = float(max_value.values)
    data[i] = pd.DataFrame(data[i]).fillna(filling_value)
#%% 
    
#Data Engeeniring
corr = data.drop('TARGET',axis=1).corr().abs()
corr_series = corr.unstack()
corr_series_sort = corr_series.sort_values(kind="quicksort",ascending=False)

corr_series_sort.dropna(inplace=True)

threshhold = 0.5
num_of_corr = 0
for i in corr_series_sort.index:
    column_tuple = i
    corr_value = corr_series_sort.loc[i]
    col0 = column_tuple[0]
    col1 = column_tuple[1]
    #print(corr_value)
    if (corr_value > threshhold) & (col0 in data.columns) & (col1 in data.columns):
        data[col0 + " + " + col1] = data[col0] + data[col1]
        data[col0 + " * " + col1] = data[col0] * data[col1]
        
        data.drop([col0,col1],axis=1,inplace=True)
        num_of_corr +=1
        if col0 in discrete_values:
            discrete_values.remove(col0)
        if col1 in discrete_values:
            discrete_values.remove(col1)
#%% 
#get_dummies
discrete_values.remove('TARGET')
temp_dummies = pd.get_dummies(data[discrete_values].astype('category'))
data.drop(discrete_values,axis=1,inplace=True)
data[temp_dummies.columns] = temp_dummies

#drop object type that not 'object'
object_columns = list(data.dtypes[data.dtypes=='object'].index)
data.drop(object_columns,axis=1,inplace=True)

#replace -INF AND -0 values
data.replace(-0, 0,inplace=True)
data.replace(-np.Inf, 0,inplace=True)
data.replace(np.Inf, 0,inplace=True)
data.fillna(0,inplace=True)
#%%        
################################################################################
#Train and Test Split
from sklearn.model_selection import train_test_split
x = data.drop('TARGET',axis=1) 
y = data['TARGET']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)


#without feature selection
df_score_filter_methods = pd.DataFrame(index = ['rf','svc','neigh'])
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = None,min_samples_split= 2,max_features = None,verbose = 0)#,min_samples_leafs = 2

df_score_filter_methods.loc['rf','no fs'] = clf_rf.fit(x_train,y_train).score(x_test,y_test)

#SVC
from sklearn.svm import SVC
clf_svc = SVC(gamma='auto')
df_score_filter_methods.loc['svc','no fs'] = clf_svc.fit(x_train,y_train).score(x_test,y_test) 

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
clf_neigh = KNeighborsClassifier(n_neighbors=3)

df_score_filter_methods.loc['neigh','no fs'] = clf_neigh.fit(x_train,y_train).score(x_test,y_test) 

#%%
#Feature Selection - Filter Methods
#Chi squared test - GenericUnivariateSelect
from sklearn.feature_selection import GenericUnivariateSelect,chi2
transformer = GenericUnivariateSelect(chi2, 'percentile', param=25)
x = x.abs()#chi test cant get negative values!
x_chi = transformer.fit_transform(x,y)

x_train, x_test, y_train, y_test = train_test_split(x_chi, y, test_size=0.33, random_state=42)

#RandomForestClassifier - with chi2
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = None,min_samples_split= 2,max_features = None,verbose = 0)
df_score_filter_methods.loc['rf','chi2'] = clf_rf.fit(x_train,y_train).score(x_test,y_test)

#SVC - with chi2
clf_svc = SVC(gamma='auto')
df_score_filter_methods.loc['svc','chi2'] = clf_svc.fit(x_train,y_train).score(x_test,y_test) 

#KNeighborsClassifier - with chi2
clf_neigh = KNeighborsClassifier(n_neighbors=3)

df_score_filter_methods.loc['neigh','chi2'] = clf_neigh.fit(x_train,y_train).score(x_test,y_test) 

print(df_score_filter_methods)
#chi2 feature selection improve only random forest
#The results are not stable - some time the unfilter results are better

#%%
#correlation coefficient scores
#correlation between features
import seaborn as sns
x = data.drop('TARGET',axis=1) 
y = data['TARGET']

#corr = x.corr()
#plt.figure(p_num)
#sns.heatmap(corr, annot=True, annot_kws={"size": 40})
#plt.suptitle("correlation between features", fontsize=40)
#p_num+=1

corr = x.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if np.abs(corr.iloc[i,j]) >= 0.1:
            if columns[j]:
                columns[j] = False
selected_columns = x.columns[columns]
x = x[selected_columns]

#correlation between filtered features
#corr = x.corr()
#plt.figure(p_num)
#sns.heatmap(corr, annot=True, annot_kws={"size": 40})
#plt.suptitle("correlation between features", fontsize=40)
#p_num+=1

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
df_score_filter_methods.loc['rf','correlation coefficient scores'] = clf_rf.fit(x_train,y_train).score(x_test,y_test)

#SVC - with correlation coefficient scores
clf_svc = SVC(gamma='auto')
df_score_filter_methods.loc['svc','correlation coefficient scores'] = clf_svc.fit(x_train,y_train).score(x_test,y_test) 

#KNeighborsClassifier - with correlation coefficient scores
clf_neigh = KNeighborsClassifier(n_neighbors=3)

df_score_filter_methods.loc['neigh','correlation coefficient scores'] = clf_neigh.fit(x_train,y_train).score(x_test,y_test) 

print(df_score_filter_methods)
#correlation coefficient scores improve Random Forest and neigh

#%%
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
df_score_filter_methods.loc['rf','Recursive Feature Elimination'] = clf_rf.fit(x_train,y_train).score(x_test,y_test)

#SVC - with correlation coefficient scores
clf_svc = SVC(gamma='auto')
df_score_filter_methods.loc['svc','Recursive Feature Elimination']  = clf_svc.fit(x_train,y_train).score(x_test,y_test) 

#KNeighborsClassifier - with correlation coefficient scores
clf_neigh = KNeighborsClassifier(n_neighbors=3)

df_score_filter_methods.loc['neigh','Recursive Feature Elimination'] = clf_neigh.fit(x_train,y_train).score(x_test,y_test) 

print(df_score_filter_methods)
#correlation coefficient scores improve Random Forest and neigh


#%%
#Embedded Methods
x = data.drop('TARGET',axis=1) 
y = data['TARGET']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

#RandomForestClassifier - with Embedded Methods​
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = 10,min_samples_split= 10,max_features = 10,verbose = 0)
df_score_filter_methods.loc['rf','Embedded Methods'] = clf_rf.fit(x_train,y_train).score(x_test,y_test)

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
clf_svc = SVC(gamma='auto',C=100)
df_score_filter_methods.loc['svc','Embedded Methods']  = clf_svc.fit(x_train,y_train).score(x_test,y_test) 

print(clf_svc.fit(x_train,y_train).score(x_test,y_test))

#KNeighborsClassifier - with Embedded Methods
clf_neigh = KNeighborsClassifier(n_neighbors=100)
df_score_filter_methods.loc['neigh','Embedded Methods'] = clf_neigh.fit(x_train,y_train).score(x_test,y_test) 

print(clf_neigh.fit(x_train,y_train).score(x_train,y_train))
print(clf_neigh.fit(x_train,y_train).score(x_test,y_test))
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
clf_rf_cv['test_score'][14]#14 index give the highest score

from sklearn.metrics import f1_score
y_pred = cross_val_predict(clf_rf, x_test, y_test, cv=14)
df_score_filter_methods.loc['rf','Embedded Methods and K-Fold Cross Validation​'] = f1_score(y_test, y_pred, average=None)[0]

#important to try cv high enough

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
y_score = clf.best_estimator_.predict_proba(x_test)

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
#class 0 threshold = 0.46
y_proba = clf.best_estimator_.predict_proba(x_test)

mask_threshold_0 = y_proba[:,0]>=0.46
y_proba[mask_threshold_0,:]=0

mask_threshold_1 = y_proba[:,1]>=0.5
y_proba[mask_threshold_1,:]=1

y_pred = y_proba[:,0]

df_score_filter_methods.loc['rf','with ROC Curve​'] = f1_score(y_test, y_pred, average=None)[0]

#%% Bootstrap Aggregating 
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(clf.best_estimator_,n_estimators = 100,max_samples=0.7, max_features=0.15,bootstrap_features=True)
df_score_filter_methods.loc['rf','with Bagging​'] = bagging.fit(x_train,y_train).score(x_test,y_test)

print(df_score_filter_methods.loc['rf','with Bagging​'])

#%% Boosting
#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf_boost = AdaBoostClassifier(clf.best_estimator_,n_estimators=500)
df_score_filter_methods.loc['rf','with AdaBoostClassifier​'] = clf_boost.fit(x_train,y_train).score(x_test,y_test)
#%%Blending
from sklearn.ensemble import VotingClassifier

clf_voting = VotingClassifier(estimators=[
        ('clf_boost', clf_boost), ('clf.best_estimator_', clf.best_estimator_), ('clf_rf', clf_rf),('clf_neigh', clf_neigh),('clf_svc', clf_svc)], voting='hard')#'soft'
    

df_score_filter_methods.loc['rf','with Blending​'] = clf_voting.fit(x_train,y_train).score(x_test,y_test)

#%% close all plot
plt.close("all")


















