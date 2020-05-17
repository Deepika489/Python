# -*- coding: utf-8 -*-
"""
"""
#importing Libraries
import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.svm import SVC

#importing data
forestfire_data = pd.read_csv("forestfires.csv")

forestfire_data.head()
columns = forestfire_data.columns
forestfire_data.describe()

#Analysing data
sns.boxplot(x="wind",y="size_category", data = forestfire_data,palette = "hls")
sns.boxplot(x="rain",y="size_category",data=forestfire_data,palette = "hls")
sns.countplot(forestfire_data['size_category']);


from sklearn.model_selection import train_test_split
train,test = train_test_split(forestfire_data,test_size = 0.25)
test.head()

train_X = train.iloc[:,2:30]
train_y = train.iloc[:,-1]
test_X  = test.iloc[:,2:30]
test_y  = test.iloc[:,-1]

# Create SVM classification object --'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
help(SVC)
# kernel = linear

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 98.46

# Kernel = poly
"""
Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

Current default is 'auto' which uses 1 / n_features, 
if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.
"""

model_poly = SVC(kernel = "poly", gamma='auto')
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 99.23

# kernel = rbf
model_rbf = SVC(kernel = "rbf", gamma = 'auto')
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 71.538


"""
So from this we ca conclude that the Poly SVM gives us the best results for this model.

"""
#importing Libraries
import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.svm import SVC

#importing data
Sal_data_train = pd.read_csv("SalaryData_Train.csv")
Sal_data_test = pd.read_csv("SalaryData_Test.csv")

#Processing Data
Sal_data_train.head()
columns = list(Sal_data_train.columns)
Sal_data_train.describe()


#Creating Categorical op variable
Sal_data_train.loc[ Sal_data_train.Salary == ' <=50K', 'SO' ] = 'Low'
Sal_data_train.loc[ Sal_data_train.Salary == ' >50K', 'SO' ] = 'High'

train_X = Sal_data_train.iloc[:,[0,1,3,4,5,6,7,8,9,10,11,12]]
train_y = Sal_data_train.iloc[:,-1]
test_X  = Sal_data_test.iloc[:,[0,1,3,4,5,6,7,8,9,10,11,12]]
test_y  = Sal_data_test.iloc[:,-1]

Sal_data_train.dtypes
train_X.dtypes
 
#Analysing data
sns.boxplot(x="age",y="Salary", data = Sal_data_train,palette = "hls")  #Either x or Y needs to be numeric
sns.boxplot(x="capitalgain",y="Salary",data=Sal_data_train,palette = "hls")
sns.countplot(Sal_data_train['Salary']);
sns.countplot(Sal_data_train['native']);


#Encoding the Independent Variable
train_X["native"] = pd.Series(np.where(train_X.native.values == ' United-States',1,0),train_X.index)
train_X["sex"] = pd.Series(np.where(train_X.sex.values == ' Male',1,0),train_X.index)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_X["workclass"] = le.fit_transform(train_X["workclass"])
train_X["maritalstatus"] = le.fit_transform(train_X["maritalstatus"])
train_X["occupation"] = le.fit_transform(train_X["occupation"])
train_X["relationship"] = le.fit_transform(train_X["relationship"])
train_X["race"] = le.fit_transform(train_X["race"])

# Create SVM classification object --'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
help(SVC)
# kernel = linear

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 98.46

# Kernel = poly
"""
Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

Current default is 'auto' which uses 1 / n_features, 
if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.
"""

model_poly = SVC(kernel = "poly", gamma='auto')
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 99.23

# kernel = rbf
model_rbf = SVC(kernel = "rbf", gamma = 'auto')
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 71.538