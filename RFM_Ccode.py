
# coding: utf-8

# In[2]:


import pandas as pd
diabetes_df=pd.read_csv(r'C:\Users\HARI\Desktop\Neelima\Reva\DataSets\diabetes.csv')


# In[3]:


import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# In[4]:


y = diabetes_df.Outcome


# In[8]:


import numpy as np

diabetes_df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]= diabetes_df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

diabetes_df.dropna(inplace=True)
y = diabetes_df.Glucose


# In[ ]:


def f(row):
   if (row["Glucose"] < 100):
       val= 'below 100'

   elif(row["Glucose"] < 120):
       val= 'below 120'

   elif(row["Glucose"] < 160):
       val = 'below 160'

   elif(row["Glucose"]< 180):
       val= 'below 180'

   elif(row["Glucose"] >= 180):
       val= 'above 180'    
   return val


# In[20]:


def fa(row):
    if (row["Age"] < 30):
        val= 'below 30'
    
    elif(row["Age"] < 40):
        val= 'below 40'
    
    elif(row["Age"] < 50):
        val = 'below 50'
    
    elif(row["Age"] >= 50):
        val= 'above 50'    
    return val


# In[21]:


diabetes_df["OutcomeNum"] = np.where(diabetes_df["Outcome"] == True,1,0)


# In[17]:


diabetes_df["OutcomeNum"] = np.if(diabetes_df["Outcome"] == "True"),1,0
df["Col3"] = np.where(df['Col2'].isin(['Z','X']), "J", np.where(df['Col2'].isin(['Y']), 'K', df['Col1']))


# In[28]:


diabetes_df["Age Groups"] =diabetes_df.apply(fa,axis=1)
diabetes_df["Gluoselevels"] =diabetes_df.apply(f,axis=1)
diabetes_df["OutcomeNum"] = np.where(diabetes_df["Outcome"] == True,1,0)


# In[22]:


diabetes_df.head()


# In[39]:


X_train, X_test, y_train, y_test = train_test_split( diabetes_df, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[31]:


X_train, X_test, y_train, y_test = train_test_split( diabetes_df, y, test_size=X_train, X_test, y_train, y_test = train_test_split( diabetes_df, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[40]:


X_train.tail()


# In[27]:


train.head()


# In[29]:


import numpy as np
train, test = np.split(diabetes_df.sample(frac=1), [int(.7*len(diabetes_df))])



# In[30]:


pd.pivot_table(train, values="Outcome", index=["Gluoselevels"],               columns="Age Groups")
# pd.pivot_table(df, values="TestScore", index=["company"],\
#           columns="regiment", fill_value=0)


# In[12]:


pd.pivot_table(test, values="Outcome", index=["Gluoselevels"],               columns="Age Groups", fill_value=0)


# In[22]:


# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


# In[23]:


predictions[0:5]


# In[24]:


## The line / model
plt.scatter(y_test, predictions,s=30)
plt.xlabel("True Values")
plt.ylabel("Predictions")


# In[19]:


print(y_test[0:10])

