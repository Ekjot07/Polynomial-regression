#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import hvplot.pandas
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')

dataset = pd.read_csv('E:\Semester 4\ML\Housing.csv')
#dataset.head()
#dataset.shape

X = dataset[['area', 'bedrooms', 'bathrooms','stories',]]
y = dataset['price']
#print(y_test.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)



from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2 = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2)
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2 = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

Xtrain2 = poly_reg.fit_transform(X_train)
Xtest2 = poly_reg.transform(X_test)

linR = LinearRegression(normalize=True)
linR.fit(Xtrain2,y_train)

testP = linR.predict(Xtest2)
trainP = linR.predict(Xtrain2)

print('Test set evaluation:\n')
print_evaluate(y_test, testP)
print('Train set evaluation:\n')
print_evaluate(y_train, trainP)

resultsDF = pd.DataFrame(data=[["Polynomail Regression", *evaluate(y_test, testP), 0]], columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

#plt.scatter(dataset['area'], dataset['price'], c='red')
#plt.scatter(dataset['bedrooms'], dataset['price'], c='red')
#plt.scatter(dataset['bathrooms'], dataset['price'], c='red')
#plt.scatter(dataset['stories'], dataset['price'], c='red')
print("Predicted values: \n")
df = pd.DataFrame(trainP)
df
#df2 = pd.DataFrame(testP)
#df2
#print(linR.intercept_)
#print(linR.coef_)

