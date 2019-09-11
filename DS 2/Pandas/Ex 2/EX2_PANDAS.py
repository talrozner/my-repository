#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True #autocomplete jupyter')


# In[5]:


#load the file
file_path = r'DatafinitiElectronicsProductsPricingData.csv'
data = pd.read_csv(file_path)

#data info()
data.info()


# In[6]:


data.head()


# In[ ]:




