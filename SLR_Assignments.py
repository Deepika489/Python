# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:54:44 2020

@author: visveshbabu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""""SLM Assignment 1

calories = pd.read_csv("C:\\DataScience\\Assignments\\Simple Linear Regression\\calories_consumed.csv")
ip_c = calories.iloc[:,0]
op_c = calories.iloc[:,1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(ip_c,op_c, test_size = 0.2, random_state = 0)

x_train = x_train.values.reshape(-1,1)
y_train = y_train.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train,y_train)

y_predict = reg.predict(x_test.values.reshape(-1,1))

#Visualizing the training set results
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,reg.predict(x_train), color = 'blue')
plt.show()

#Visualizing the test set results
plt.scatter(x_test,y_test,color = 'green')
plt.plot(x_train,reg.predict(x_train), color = 'pink')
plt.show()

End """""


delivery_time = pd.read_csv("C:\\DataScience\\Assignments\\Simple Linear Regression\\delivery_time.csv")
ip_d = delivery_time.iloc[:,-1]
op_d = delivery_time.iloc[:,1]

delivery_time.corr()
np.corrcoef(ip_d,op_d)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(ip_d,op_d, test_size = 1/3, random_state = 0)

x_train = x_train.values.reshape(-1,1)
y_train = y_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train,y_train)

y_predict = reg.predict(x_test)

op = pd.DataFrame(data = y_predict)
op.corr(ip_d)



churn_out_rate = pd.read_csv("C:\\DataScience\\Assignments\\Simple Linear Regression\\emp_data.csv")
ip_o = churn_out_rate.iloc[:,-1]
op_o = churn_out_rate.iloc[:,1]

salary_hike = pd.read_csv("C:\\DataScience\\Assignments\\Simple Linear Regression\\Salary_Data.csv")
ip_s = salary_hike.iloc[:,-1]
op_s = salary_hike.iloc[:,1]


#1) Calories_consumed-> predict weight gained using calories consumed
#2) Delivery_time -> Predict delivery time using sorting time 
#3) Emp_data -> Build a prediction model for Churn_out_rate 
#4) Salary_hike -> Build a prediction model for Salary_hike
