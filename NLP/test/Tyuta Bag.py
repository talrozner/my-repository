'''
from collections import Counter
def my_counter(x):
#data['pos tag'].str.split(' ')
    temp_features = Counter([j for j in x])#data['pos tag'].str.split(' ')[0]
    return temp_features


temp_features = data['pos tag'].str.split(' ').apply(lambda x: my_counter(x))
temp_features = pd.DataFrame.from_dict(temp_features,orient = 'index')

temp_features = vectorizer.fit_transform(data['pos dependency'])
train_data_features = np.append(train_data_features,temp_features,axis=1)

temp_features = vectorizer.fit_transform(data['dep dependency'])
train_data_features = np.append(train_data_features,temp_features,axis=1)
'''