# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:38:24 2020

@author: visveshbabu 18001030344
"""

"""
Prepare a prediction model for profit of 50_startups data.
Do transformations for getting better predictions of profit and
make a table containing R^2 value for each prepared model.

R&D Spend -- Research and devolop spend in the past few years
Administration -- spend on administration in the past few years
Marketing Spend -- spend on Marketing in the past few years
State -- states from which data is collected
Profit  -- profit of each state in the past few years
"""

import pandas as pd
# Reading the input as a Data Frame
dataset = pd.read_csv("C:\\DataScience\\Assignments\\Multi Linear Regression\\50_Startups.csv")

#op is the Profit and the ip are R&D Spend, Administration,Marketing Spend,State.
# As the state has categorical values converting into continuous data using one hot encoder
ip = dataset.iloc[:,:4].values
op = dataset.iloc[:,4].values

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder = LabelEncoder()
ip[:,3] = labelencoder.fit_transform(ip[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
ip = onehotencoder.fit_transform(ip).toarray()
