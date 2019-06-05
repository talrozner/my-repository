#%%Home Credit Default Risk
#Home Credit Default Risk
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

#load data
file_path_0 = r'D:\DS\Seminars\ML Pipeline\Home Credit Default Risk\csv\data\sample_tr_0.csv'
file_path_1 = r'D:\DS\Seminars\ML Pipeline\Home Credit Default Risk\csv\data\sample_tr_1.csv'

data = pd.read_csv(file_path_0)

data = data.append(pd.read_csv(file_path_1))
data.reset_index(inplace = True)
data.drop('index',axis = 1,inplace = True)

#%%Type of Features
float64_columns = list(data.dtypes[data.dtypes=='float64'].index)
int64_columns = list(data.dtypes[data.dtypes=='int64'].index)
object_columns = list(data.dtypes[data.dtypes=='object'].index)

#%%Continuous values Vs Discrete values
continuous_values = list()
discrete_values = list()
for i in data.columns:
    if len(data[i].unique()) > 20:
        continuous_values.append(i)    
    else:
        discrete_values.append(i)
        
#new_data = data.iloc[:,:2]
#new_data[object_columns[:12]]= data[object_columns[:12]]
#new_data[discrete_values[:12]] = data[discrete_values[:12]]
#new_data[continuous_values[:12]] = data[continuous_values[:12]]
#data = new_data

#object_columns = object_columns[:12]
#discrete_values = discrete_values[:12]
#continuous_values = continuous_values[:12]
        
#################################################################################
#%%Data - Info
data.info()
data.dtypes

#%%Type of Features
float64_columns = list(data.dtypes[data.dtypes=='float64'].index)
int64_columns = list(data.dtypes[data.dtypes=='int64'].index)
object_columns = list(data.dtypes[data.dtypes=='object'].index)

#%%filling na
isna_label = data['TARGET'].isna().any()
columns_na = data.isna().any()

data.fillna(0,inplace = True)

#drop 'FLAG_MOBIL'
data.drop('FLAG_MOBIL',axis=1,inplace=True)



#%%get_dummies
discrete_values.remove('TARGET')
discrete_values.remove('FLAG_MOBIL')
temp_dummies = pd.get_dummies(data[discrete_values].astype('category'))
data.drop(discrete_values,axis=1,inplace=True)
data[temp_dummies.columns] = temp_dummies

object_continues_columns = list(set(object_columns) - set(discrete_values))
temp_dummies = pd.get_dummies(data[object_continues_columns].astype('category'))
data.drop(object_continues_columns,axis=1,inplace=True)
data[temp_dummies.columns] = temp_dummies

#%%scale negative values
temp = data.min()
mask = list(temp.values<0)
col_mask = data.columns[mask]

data[col_mask] = data[col_mask] + data[col_mask].min().abs()

#fill inf values
data.replace([np.inf, -np.inf], 0,inplace=True)

data.drop(data.columns[data.isna().any()],axis=1,inplace=True)

#Train and Test Split
from sklearn.model_selection import train_test_split
x = data.drop('TARGET',axis=1) 
y = data['TARGET']

#x = (x - x.min())/(x.max()-x.min())
#x.columns[x.isna().any()]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

#%%Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500)
clf.fit(x_train, y_train)

y_pred_train = clf.predict(x_train)
f1_score_final_train = f1_score(y_train,y_pred_train)  

y_pred_test = clf.predict(x_test)
f1_score_final_test = f1_score(y_test,y_pred_test)  

print("f1 score of train randomforest " + str(f1_score_final_train))
print("f1 score of test randomforest " + str(f1_score_final_test))

#%%XGBoost
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=1,min_samples_split=50,min_samples_leaf=50).fit(x_train, y_train)

y_pred_train = clf.predict(x_train)
f1_score_final_train = f1_score(y_train,y_pred_train)  

y_pred_test = clf.predict(x_test)
f1_score_final_test = f1_score(y_test,y_pred_test)  

print("f1 score of train XGBoost " + str(f1_score_final_train))
print("f1 score of test XGBoost " + str(f1_score_final_test))

#%%finding indication
indication = x_train.columns[x_train.max()==1]
#only_zaroes = x_train.columns[x_train.max()==0]
#only_zaroes = only_zaroes & x_train.min()==0
#%%AutoEncoders - V1
import os

from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

x_train_np = np.copy(np.array(x_train[indication]))
x_test_np = np.copy(np.array(x_test[indication]))

# define the number of features
ncol = x_train_np.shape[1]

#Design Auto Encoder
### Define the encoder dimension
encoding_dim = 100#5

input_dim = Input(shape = (ncol, ))

# Encoder Layers
encoded1 = Dense(500, activation = 'relu')(input_dim)
encoded2 = Dense(400, activation = 'relu')(encoded1)
encoded3 = Dense(300, activation = 'relu')(encoded2)
encoded4 = Dense(200, activation = 'relu')(encoded3)
encoded5 = Dense(150, activation = 'relu')(encoded4)
encoded6 = Dense(125, activation = 'relu')(encoded5)
encoded7 = Dense(encoding_dim, activation = 'relu')(encoded6)

# Decoder Layers
decoded1 = Dense(125, activation = 'relu')(encoded7)
decoded2 = Dense(150, activation = 'relu')(decoded1)
decoded3 = Dense(200, activation = 'relu')(decoded2)
decoded4 = Dense(300, activation = 'relu')(decoded3)
decoded5 = Dense(400, activation = 'relu')(decoded4)
decoded6 = Dense(500, activation = 'relu')(decoded5)
decoded7 = Dense(ncol, activation = 'sigmoid')(decoded6)

# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_dim, outputs = decoded7)

# Compile the Model
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

#autoencoder.summary()

#Train Auto Encoder
autoencoder.fit(x_train_np, x_train_np, nb_epoch = 10, batch_size = 32, shuffle = False, validation_data = (x_test_np, x_test_np))

#Use Encoder level to reduce dimension of train and test data
encoder = Model(inputs = input_dim, outputs = encoded7)
encoded_input = Input(shape = (encoding_dim, ))

#Predict the new train and test data using Encoder
encoded_train = pd.DataFrame(encoder.predict(x_train_np))
encoded_train = encoded_train.add_prefix('feature_')

encoded_test = pd.DataFrame(encoder.predict(x_test_np))
encoded_test = encoded_test.add_prefix('feature_')


#%%XGBoost on encoded features
x_train_new = x_train.drop(indication,axis=1)
x_train_new = x_train_new.join(encoded_train)
x_train_new.fillna(0,inplace=True)

x_test_new = x_test.drop(indication,axis=1)
x_test_new = x_test_new.join(encoded_test)
x_test_new.fillna(0,inplace=True)

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=1,min_samples_split=50,min_samples_leaf=50).fit(x_train_new, y_train)

y_pred_train = clf.predict(x_train_new)
f1_score_final_train = f1_score(y_train,y_pred_train)  

y_pred_test = clf.predict(x_test_new)
f1_score_final_test = f1_score(y_test,y_pred_test)  

print("f1 score of train XGBoost on Encoded features " + str(f1_score_final_train))
print("f1 score of test XGBoost on Encoded features " + str(f1_score_final_test))





#%% close all plot
plt.close("all")