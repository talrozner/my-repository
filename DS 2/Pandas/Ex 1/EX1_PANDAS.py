#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True #autocomplete jupyter')


# In[5]:


#load the file
file_path = r'AB_NYC_2019.csv'
data = pd.read_csv(file_path)

#data info()
data.info()


# In[20]:


#find the most common neighbourhood
mask = data['neighbourhood'].value_counts() > 300
sort_neighbourhood = data['neighbourhood'].value_counts()[mask]
sort_neighbourhood = sort_neighbourhood.index


# In[63]:


#calculate average price
def is_in_sort_neighbourhood(x):
    if x in sort_neighbourhood:
        return True
    else:
        return False
    
data["is in neighbourhood"] = data['neighbourhood'].apply(is_in_sort_neighbourhood).values

mask = data["is in neighbourhood"] == True

average_price =data.loc[mask,'price'].mean()


# In[92]:


#apartments below the average price
mask = (data["is in neighbourhood"] == True) & (data['price']<average_price)
data[mask].shape


# In[93]:


#Calculate the distance of flats from the beginning of the axes
data['radius'] = np.sqrt(data['latitude']**2 + data['longitude']**2)


# In[95]:


# 10 moset close to axis and apartments below the average price
final_data = data[mask].sort_values('radius')[:10]


# In[101]:


#Drop the apartments that don’t have review
mask = final_data['reviews_per_month'].isnull().values
final_data[mask]


# In[ ]:




