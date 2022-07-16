#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


# In[4]:


hr=pd.read_csv(r"E:\HRAnalytics\HR Analytics\project_newcolumns.csv")


# In[5]:


hr.shape


# In[6]:


hr.head()


# In[7]:


for column in hr.columns:
    hr[column]=hr[column].replace({'-':np.nan})
    print(str(column)+" : " + 'null value present')
    print("---------------------------------------------------------------")
    


# In[8]:


hr=hr.drop(['Termination Date'],axis=1)


# In[9]:


hr.head()


# In[10]:


hr.shape


# In[11]:


hr.isnull().sum()


# In[12]:


hr['Utilization%']=hr['Utilization%'].str.replace('%',' ')


# In[13]:


hr.info()


# In[14]:


hr['Current Status'].replace({'Active':0,'Resigned':1},inplace=True)


# In[15]:


hr.info()


# In[16]:


name=hr['Employee Name']


# In[17]:


hr=hr.drop(['Employee Name'],axis=1)


# In[18]:


hr=hr.drop(['Employee No'],axis=1)


# In[19]:


hr.head()


# In[20]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[21]:


hr[hr.select_dtypes(include=['object']).columns] = hr[hr.select_dtypes(include=['object']).columns].apply(le.fit_transform)


# In[22]:


hr.head()


# In[23]:


hr.corr()


# In[24]:


plt.figure(figsize=(15,15))
sns.heatmap(hr.corr(),annot=True,cmap='coolwarm',linecolor='white')


# In[25]:


hr.head()


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


y=hr['Current Status']


# In[28]:


x=hr.drop(['Current Status'],axis=1)


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=111)


# In[30]:


x_train.shape


# In[31]:


y_train.shape


# In[32]:


x_test.shape


# In[33]:


y_test.shape


# In[39]:


x_train


# In[35]:


y_train.head()


# In[36]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[37]:


chi_test_hr=SelectKBest(score_func=chi2,k='all')


# In[40]:


fitted_hr=chi_test_hr.fit(abs(x),y)


# In[41]:


fitted_hr.scores_


# In[43]:


list(fitted_hr.scores_)


# In[44]:


df1_hr=pd.DataFrame({'Feature':x.columns,'Importance':fitted_hr.scores_})


# In[45]:


df1_hr.sort_values('Importance',ascending=False)


# In[53]:


#RFE


# In[54]:


from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier


# In[55]:


dt=DecisionTreeClassifier()


# In[56]:


rfe_hr=RFE(dt,18)


# In[57]:


rfe_hr.fit(x,y)


# In[58]:


rfe_hr.support_


# In[59]:


#Boruta


# In[60]:


from boruta import BorutaPy


# In[61]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[62]:


hr1=x


# In[63]:


hr_x=np.array(x)
hr_y=np.array(y)


# In[64]:


boruta_feature_selector=BorutaPy(rf,max_iter=30,verbose=2,random_state=555)


# In[65]:


boruta_feature_selector.fit(hr_x,hr_y)


# In[66]:


boruta_feature_selector.support_


# In[69]:


df2_hr=pd.DataFrame({'Feature':hr1.columns,'Importance':boruta_feature_selector.support_})


# In[68]:


df2_hr.sort_values('Importance',ascending=False)


# In[ ]:




