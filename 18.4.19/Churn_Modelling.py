#%%Churn_Modelling
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def my_f1_score(y_test, y_pred):
    f1_score_ = f1_score(y_test, y_pred, average='weighted')#[0]
    return f1_score_
    
p_num=1

    
#load data
file_path = r"D:\DS\Seminars\ML Pipeline\Churn Modelling\Churn_Modelling.csv"

data = pd.read_csv(file_path)
data.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)

#%%Fill na
columns_na = data.columns[data.isna().any()]
#no missing values
#%% 
#get_dummies
#'Geography' 'Gender'
temp_dummies = pd.get_dummies(data[['Geography','Gender']].astype('category'))
data.drop(['Geography','Gender'],axis=1,inplace=True)
data[temp_dummies.columns] = temp_dummies

#%%Data Exploration
#histogram
hist_list = ['CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary','Exited']
hist_list = list(data.columns)
for i in hist_list:
#i = 'Gender'
    column_name = i#'CreditScore'
    
    mask_0 = data['Exited']==0
    feature_0 = data.loc[mask_0,column_name]
    mask_1 = data['Exited']==1
    feature_1 = data.loc[mask_1,column_name]
    
    plt.figure(p_num)
    plt.hist(feature_0,stacked=True, histtype='bar',normed=True)
    plt.hist(feature_1,stacked=True, histtype='bar',normed=True)
    plt.xlabel(column_name,fontdict={'fontsize': 40})
    plt.ylabel("Value",fontdict={'fontsize': 40})
    plt.legend(['label 0','label 1'],prop={'size': 30})
    p_num +=1

# %%Train and Test Split
from sklearn.model_selection import train_test_split
x = data.drop('Exited',axis=1) 
y = data['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)


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

#%%Feature Selection - Filter Methods
#Chi squared test - GenericUnivariateSelect
from sklearn.feature_selection import GenericUnivariateSelect,chi2
transformer = GenericUnivariateSelect(chi2, 'percentile', param=1)
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

#%%correlation coefficient scores
#correlation between features
import seaborn as sns
x = data.drop('Exited',axis=1) 
y = data['Exited']

#corr = x.corr()
#plt.figure(p_num)
#sns.heatmap(corr, annot=True, annot_kws={"size": 40})
#plt.suptitle("correlation between features", fontsize=40)
#p_num+=1

corr = x.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if np.abs(corr.iloc[i,j]) >= 0.9:
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
#corr_label = x.corr()['Exited'].to_frame()
#corr_label.sort_values('Exited',inplace=True)
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
#%% Wrapper Methods
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
#%% Embedded Methods
x = data.drop('Exited',axis=1) 
y = data['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

#RandomForestClassifier - with Embedded Methods​
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = 10,min_samples_split= 10,max_features = 10,verbose = 0)
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
clf_neigh = KNeighborsClassifier(n_neighbors=100)
y_pred = clf_neigh.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['neigh','Embedded Methods'] = my_f1_score(y_test, y_pred)




#%% ####################################################################
#for this part i will continue only with random forest
#K-Fold Cross Validation​
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

x = data.drop('Exited',axis=1) 
y = data['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

#RandomForestClassifier - with Embedded Methods​ and K-Fold Cross Validation​
clf_rf = RandomForestClassifier(n_estimators = 500,max_depth = 10,min_samples_split= 10,max_features = 10,verbose = 0)

clf_rf_cv = cross_validate(clf_rf, x_train, y_train, cv=30, scoring='f1_weighted',return_estimator=True)

clf_rf_cv['test_score']
clf_rf_cv['test_score'][11]#11 index give the highest score

y_pred = cross_val_predict(clf_rf, x_test, y_test, cv=11)
df_score_filter_methods.loc['rf','Embedded Methods and K-Fold Cross Validation​'] = my_f1_score(y_test, y_pred)
#important to try cv high enough


#%% ROC curve AUC
'''
#ROC curve AUC
from sklearn.metrics import roc_curve, auc
y_score = cross_val_predict(clf_rf, x_test, y_test, cv=11,method='predict_proba')

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

#class 0 threshold = 0.65
#class 1 threshold = 0.13
y_proba = y_score

mask_threshold_0 = y_proba[:,0]>=0.65
y_proba[mask_threshold_0,:]=0

mask_threshold_1 = y_proba[:,1]>=0.13
y_proba[mask_threshold_1,:]=1

y_pred = y_proba[:,0]

df_score_filter_methods.loc['rf','with ROC Curve​'] = my_f1_score(y_test, y_pred)
'''
#%% Bootstrap Aggregating 
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(clf_rf_cv['estimator'][0],n_estimators = 100,max_samples=0.7 ,bootstrap_features=True)#max_features=0.5
y_pred = bagging.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['rf','with Bagging'] = my_f1_score(y_test, y_pred)



#%% Boosting
#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf_boost = AdaBoostClassifier(clf_rf_cv['estimator'][0],n_estimators=500)
y_pred = clf_boost.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['rf','AdaBoostClassifier'] = my_f1_score(y_test, y_pred)

#%%Blending
from sklearn.ensemble import VotingClassifier

clf_voting = VotingClassifier(estimators=[('clf_boost', clf_boost), ('clf_rf_cv', clf_rf_cv['estimator'][0]), ('clf_rf', clf_rf),('clf_neigh', clf_neigh),('clf_svc', clf_svc)], voting='hard')#'soft'
    

y_pred = clf_voting.fit(x_train,y_train).predict(x_test)
df_score_filter_methods.loc['rf','Blending'] = my_f1_score(y_test, y_pred)

#%% close all plot
plt.close("all")



















