# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:42:47 2020

@author: visveshbabu
"""
"""
1) Prepare a classification model using Naive Bayes 
for salary data 

Data Description:

age -- age of a person
workclass	-- A work class is a grouping of work 
education	-- Education of an individuals	
maritalstatus -- Marital status of an individulas	
occupation	 -- occupation of an individuals
relationship -- 	ccc
race --  Race of an Individual
sex --  Gender of an Individual
capitalgain --  profit received from the sale of an investment	
capitalloss	-- A decrease in the value of a capital asset
hoursperweek -- number of hours work per week	
native -- Native of an individual
Salary -- salary of an individual
"""

#importing the libraries
import numpy as np             #general-purpose array-processing package.
import pandas as pd            #data analysis and manipulation
import matplotlib.pyplot as plt #Visualisation

#importing the dataset
dataset_train = pd.read_csv("C:\\DataScience\\Assignments\\Naive Bayes\\SalaryData_Train.csv")
dataset_test = pd.read_csv("C:\\DataScience\\Assignments\\Naive Bayes\\SalaryData_Test.csv")

X_train =
Y_train = 
X_test = 
Y_test = 

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
 salary_train = number.fit_transform(salary_train[i])
 salary_test = number.fit_transform(salary_test[i])


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

ignb = GaussianNB()
imnb = MultinomialNB()

# Building and predicting at the same time 

pred_gnb = ignb.fit(Xtrain,ytrain).predict(Xtest)
pred_mnb = imnb.fit(Xtrain,ytrain).predict(Xtest)


# Confusion matrix GaussianNB model
confusion_matrix(ytest,pred_gnb) # GaussianNB model
pd.crosstab(ytest.values.flatten(),pred_gnb) # confusion matrix using 
np.mean(pred_gnb==ytest.values.flatten()) # 100%


sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 80%

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print("Accuracy",(10891+780)/(10891+780+2920+780))  # 75%


*-6GG9tggggggggggggggtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt

################## Reading the Salary Data 
salary_train = pd.read_csv("E:\\Bokey\\Excelr Data\\Python Codes\\all_py\\Naive Bayes\\SalaryData_Train.csv")
salary_test = pd.read_csv("E:\\Bokey\\Excelr Data\\Python Codes\\all_py\\Naive Bayes\\SalaryData_Test.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 80%

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print("Accuracy",(10891+780)/(10891+780+2920+780))  # 75%

#########################################################################



"""
Build a naive Bayes model on the data set for classifying the ham and spam
"""
#importing the libraries
import numpy as np             #general-purpose array-processing package.
import pandas as pd            #data analysis and manipulation
import matplotlib.pyplot as plt #Visualisation


Ham_dataset  = pd.read_csv("C:\\DataScience\\Assignments\\Naive Bayes\\sms_raw_NB.csv", encoding='latin-1')

import re

#Cleaning the data
corpus = []
for i in range(0,5559):
    sh_text = Ham_dataset['text'][i]
    sh_text = re.sub("[^A-Za-z" "]+"," ",Ham_dataset['text'][i]).lower()
    sh_text = sh_text.split()
    w = []
    for word in sh_text:
        if len(word) > 3:
            w.append(word) 
    sh_text = ' '.join(w)  
    corpus.append(sh_text)          
        
Ham_dataset.loc[:,'typenum'] = Ham_dataset.type.map(dict(ham=1,spam = 0))


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = Ham_dataset.iloc[:, 2].values


# Preparing a naive bayes model on data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(X,y)
y_pred = classifier_mb.predict(X)
accuracy_m = np.mean(y == y_pred)  # 98%

# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(X,y)
ygb_pred = classifier_gb.predict(X)
accuracy_gb = np.mean(y == ygb_pred)  # 89.67% 
########################################


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_orig = cv.fit_transform(Ham_dataset.iloc[:,1]).toarray()



# Preparing a naive bayes model on data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(X_orig,y)
yorig_pred = classifier_mb.predict(X_orig)
accuracy_orig = np.mean(y == yorig_pred)  # 99%


# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(X_orig,y)
ygbo_pred = classifier_gb.predict(X_orig)
accuracy_gbo = np.mean(y == ygbo_pred)  # 93.75% 
########################################