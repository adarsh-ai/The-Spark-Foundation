#!/usr/bin/env python
# coding: utf-8

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

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the dataset

# In[25]:


df = pd.read_csv("http://bit.ly/w-data")
df


# In[26]:


df.plot.scatter('Hours','Scores')


# We see there is linear relationship between hours of study and score obtained.

# In[27]:


df.corr()


# Correlation score is 0.976

# ## Regression model

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error


# In[29]:


X = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# In[30]:


def regression(model, X, y, split):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=split,
                                                        random_state=0)

    print('Training features shape :', X_train.shape)
    print('Testing features shape : ', X_test.shape)
    print('Training output shape : ', y_train.shape)
    print('Testing output shape : ', y_test.shape)
    print()

    model.fit(X_train, y_train)

    a = model.coef_[0]
    b = model.intercept_

    print("Slope of fitted line :", a)
    print("Intercept of fitted line :", b)
    print()

    y_pred = model.predict(X_train)

    print("RMSE of fit on training data:",
          np.sqrt(mean_squared_error(y_train, y_pred)))
    print("R^2 score of fit on training data :", r2_score(y_train, y_pred))
    print()

    y_pred = model.predict(X_test)

    print("RMSE of fit on test data:",
          np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R^2 score of fit on test data :", r2_score(y_test, y_pred))

    plt.scatter(X_train, y_train, c='blue')
    plt.scatter(X_test, y_test, c='red')
    plt.legend(['Train', 'Test'])
    plt.xlabel('Hours')
    plt.ylabel('Scores')
    x1, x2 = plt.xlim()

    x = np.linspace(x1, x2, 100)
    y = a * x + b

    plt.plot(x, y)

    return model


# In[31]:


model = regression(LinearRegression(), X, y, 0.25)


# ## What will be predicted score if a student studies for 9.25 hrs/ day?

# In[32]:


input_hour = 9.25
predicted_score = model.predict(np.array(input_hour).reshape(-1, 1))[0]

print('Predicted score for a student studying %.2f hours : %.2f' %
      (input_hour, predicted_score))


# ##### According to the regression model if a student studies 9.25 hours a day, he/she is likely to score 93.89 marks

# With $R^2$ score of 0.937 and RMSE of 4.509 on testing data, we can conclude our model fits data well. As there is just simple regression, there is no room of improvement by using regularization like lasso, ridge or elastic-net. If you want to experiment using different models, pass model into regression function.
# 
# # Thank You :) 
