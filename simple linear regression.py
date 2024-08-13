#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sms
import seaborn as sns
sns.set()


# In[7]:


data=pd.read_csv('C:\\Users\\grish\\grishma python codes\\abcd.csv')


# In[8]:


data.head()


# In[9]:


data.describe()


# In[12]:


x1=data['SAT']
y=data['GPA']


# In[30]:


plt.scatter(x1,y,c='purple')
plt.title('Student performance',fontsize=20)
plt.xlabel('SAT',fontsize=15)
plt.ylabel('GPA',fontsize=15)
plt.show()


# In[26]:


x=sms.add_constant(x1)
result=sms.OLS(y,x).fit()
result.summary()


# In[29]:


plt.scatter(x1,y,c='green')
plt.title('STUDENTS MARKS',fontsize=30)
yhat=0.0017*x1+0.2750
fig=plt.plot(x1,yhat,lw=3,c='black')
plt.show()


# In[ ]:




