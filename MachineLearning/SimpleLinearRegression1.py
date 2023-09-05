# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:05:23 2023

@author: PRANABESH DEY
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


from sklearn import datasets, linear_model


# In[5]:


from sklearn.metrics import mean_squared_error


# In[6]:


diabetes = datasets.load_diabetes()


# In[7]:


#print(diabetes.keys())


# In[8]:


diabetes


# In[9]:


diabetes_x = diabetes.data[:50,2,np.newaxis]
diabetes_x


# In[10]:


diabetes_y = diabetes.target[:50,np.newaxis]
diabetes_y


# ### Newaxis - used to increase the dimension of the existing array by one more dimension, when used once. Thus, 1D array will become 2D array. 2D array will become 3D array

# In[11]:


len(diabetes_x)


# In[12]:


diabetes_x_train = diabetes_x[:-30]


# ![Screenshot%202023-08-20%20203256.png](attachment:Screenshot%202023-08-20%20203256.png)
# from 0 to -30th index

# In[13]:


diabetes_x_test = diabetes_x[-30:]


# ![Screenshot%202023-08-20%20203618.png](attachment:Screenshot%202023-08-20%20203618.png)

# In[14]:


diabetes_y_train = diabetes_y[:-30]


# In[15]:


diabetes_y_test = diabetes_y[-30:]


# In[16]:


model = linear_model.LinearRegression()


# In[17]:


model.fit(diabetes_x_train,diabetes_y_train)


# In[18]:


diabetes_y_predict = model.predict(diabetes_x_test)


# In[19]:


print("Sum of suqared error - ",mean_squared_error(diabetes_y_test,diabetes_y_predict))


# In[24]:


print("Weights : ",model.coef_)


# In[25]:


print("Intercept :",model.intercept_)


# In[26]:


plt.scatter(diabetes_x_test,diabetes_y_test)


# In[30]:


plt.plot(diabetes_x_test,diabetes_y_predict)


# In[31]:


plt.show()


# In[ ]:




