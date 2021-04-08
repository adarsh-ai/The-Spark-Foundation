# # Author : Adarsh Ravankar
#
# # Task 1 : Prediction using Supervised Machine Learning
#
# # GRIP @ The Sparks Foundation
#
# ## Problem Statement :
#
# To predict percentage of marks that student expected to score based on no. of hours of studying.
#
# We would use linear regression where independent variable is no. of hours and dependent variable is score.
#
# ## Importing libraries
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading data from url
url="http://bit.ly/w-data"
Df=pd.read_csv(url)
Df.head(10)


# In[3]:


# Plotting the distribution of scores
Df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[4]:


X = Df.iloc[:, :-1].values  
y = Df.iloc[:, 1].values


# In[5]:


#split this data into training and test sets
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# In[6]:


#Training the algorithm
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[7]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[8]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[9]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[10]:


np.shape(Df)


# In[11]:


#make predict score if student study in 9.25 hours/days
hours=9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[12]:


#check mean square error
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




