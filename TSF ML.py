#!/usr/bin/env python
# coding: utf-8

# <h1>Prediction using Supervised ML</h1>

# In[2]:


pip install pandas


# In[3]:


pip install numpy


# In[4]:


pip install matplotlib


# In[10]:


import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt


# <h3>Reading Data</h3>

# In[11]:


data = pd.read_csv("scores.csv")  
print("Data read!")
data.head()


# In[12]:


data.plot(x='Hours', y='Scores', style='*')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# <h3>Prearing Data</h3>

# In[13]:


x = data.iloc[:,:1].values    
y = data.iloc[:, 1].values  


# In[14]:


print("x= ",x)
print("y= ",y)


# <h3>Training ML</h3>

# In[15]:


from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)  


# In[16]:


from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(x_train, y_train)   


# In[17]:


line = regressor.coef_*x+regressor.intercept_  
plt.scatter(x,y)  
plt.plot(x, line);  
plt.show()  


# <h3>Testing</h3>

# In[18]:


hours=0
while hours!=[[-1]]:
    hours = [[float(input("Enter number of hours studied(Enter '-1' to stop):"))]]
    if hours == [[-1]]:
        break
    pred = regressor.predict(hours)
    if pred[0]>100:
        pred[0]=100
    print("Number of hours = {}".format(hours[0][0]))
    print("Prediction Score = {}".format(pred[0]))


# In[ ]:




