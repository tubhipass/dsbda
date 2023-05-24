#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


data = pd.read_csv("titanic_train.csv")
data.head(2)


# In[5]:


from seaborn import load_dataset
tips = load_dataset("tips")


# In[6]:


sns.countplot(data['Survived'])
plt.show()


# In[7]:


data['Sex'].value_counts().plot(kind="pie",autopct="%.2f")
plt.show()


# In[8]:


data=data.dropna()
plt.hist(data['Age'],bins=5)
plt.show()


# In[9]:


sns.distplot(data['Age'])
plt.show()


# In[13]:


sns.scatterplot(x=tips["total_bill"],y=tips["tip"])
plt.show()


# In[14]:


sns.scatterplot(x=tips["total_bill"],y=tips["tip"],hue=tips["sex"])
plt.show()


# In[17]:


sns.scatterplot(x=tips["total_bill"],y=tips["tip"],hue=tips["sex"],style=tips['smoker'])
plt.show()


# In[19]:


sns.barplot(x=data['Pclass'],y=data['Age'])
plt.show()


# In[21]:


sns.barplot(x=data['Pclass'],y=data['Fare'],hue=data["Sex"])
plt.show()


# In[22]:


sns.boxplot(x=data['Sex'],y=data["Age"])
plt.show()


# In[24]:


sns.boxplot(x=data['Sex'],y=data["Age"],hue =data["Survived"])
plt.show()


# In[25]:


sns.distplot(data[data['Survived'] == 0]['Age'],hist=False, color="blue")
sns.distplot(data[data['Survived'] == 1]['Age'],hist=False, color="orange")
plt.show()


# In[26]:


pd.crosstab(data['Pclass'],data['Survived'])


# In[27]:


sns.heatmap(pd.crosstab(data['Pclass'],data['Survived']))
plt.show()


# In[28]:


sns.clustermap(pd.crosstab(data['Parch'],data['Survived']))
plt.show()


# In[ ]:




