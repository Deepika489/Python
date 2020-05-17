# -*- coding: utf-8 -*-
"""
affairs : numeric. How often engaged in extramarital sexual intercourse during the past year?
gender : factor indicating gender.
age : numeric variable coding age in years: 17.5 = under 20, 22 = 20–24, 27 = 25–29, 32 = 30–34, 37 = 35–39, 42 = 40–44, 47 = 45–49, 52 = 50–54, 57 = 55 or over.
yearsmarried : numeric variable coding number of years married: 0.125 = 3 months or less, 0.417 = 4–6 months, 0.75 = 6 months–1 year, 1.5 = 1–2 years, 4 = 3–5 years, 7 = 6–8 years, 10 = 9–11 years, 15 = 12 or more years.
children : factor. Are there children in the marriage?
religiousness : numeric variable coding religiousness: 1 = anti, 2 = not at all, 3 = slightly, 4 = somewhat, 5 = very.
education : numeric variable coding level of education: 9 = grade school, 12 = high school graduate, 14 = some college, 16 = college graduate, 17 = some graduate work, 18 = master's degree, 20 = Ph.D., M.D., or other advanced degree.
occupation : numeric variable coding occupation according to Hollingshead classification (reverse numbering).
rating : numeric variable coding self rating of marriage: 1 = very unhappy, 2 = somewhat unhappy, 3 = average, 4 = happier than average, 5 = very happy.

"""

# Importing Necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


affair = pd.read_csv('C:\\DataScience\\Assignments\\Logistic Regression\\affairs.csv')


# Getting the barplot for the categorical columns 
affair.shape
pd.crosstab(affair.affairs,affair.gender).plot(kind="bar")
pd.crosstab(affair.affairs,affair.age).plot(kind="bar")
pd.crosstab(affair.affairs,affair.yearsmarried).plot(kind="bar")
pd.crosstab(affair.affairs,affair.rating).plot(kind = "box")
pd.crosstab(affair.affairs,affair.rating).plot(kind = "bar")

#Using Seaborn getting the plots
sns.countplot(affair.rating,palette = "hls")
sns.countplot(x='age',data = affair,palette = "hls")
sns.countplot(x = 'children', data = affair, palette = "hls")
sns.countplot(affair.education)

# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns
sns.boxplot(x='age',y ='affairs', data = affair,palette = "hls")
sns.boxplot(x='affairs',y ='children', data = affair,palette = "hls")
sns.boxplot(x='religiousness',y ='affairs', data = affair,palette = "hls")

#Preprocessing
affair['gender'].fillna(0,inplace=True) # affair.gender.mode() = 0
affair['age'].fillna(32,inplace=True) # affair.age.median() = 32
affair['yearsmarried'].fillna(0,inplace=True) # affair.yearsmarried.mode() = 0
affair['children'].fillna('no',inplace=True) # affair.children.mode() = 0
affair['religiousness'].fillna(4,inplace=True) # affair.religiousness.mode() = 0
affair['occupation'].fillna(5,inplace=True) # affair.occupation.mode() = 5
affair['rating'].fillna(0,inplace=True) # affair.rating.mode() = 0
#gender,age,yearsmarried,childres,religiousness,occupation,rating

#Creating Dummy columns for Categorical Variables
affair_Imput = affair.iloc[:,[1,2,3,4,5,6,7,8]]

affair_Imput.loc[:,'gendernum'] = pd.Series(np.where(affair_Imput.gender.values == 'male',1,0),affair_Imput.index)

affair_Imput.loc[:,'childnum'] = affair_Imput.children.map(dict(yes=1,no = 0))

####affair_Imput.drop([2],inplace =True,axis = 1)
""""
pd.Series(map(lambda x: dict(yes=1, no=0)[x],
              sample.housing.values.tolist()), sample.index)
"""""
affair_Imput.head()

# Model building 
from sklearn.linear_model import LogisticRegression

X = affair_Imput.iloc[:,[1,2,4,5,6,7,8,9]]
#Y = affair.iloc[:,0]               # Series Object created
Y = affair.iloc[:,0].values
classifier = LogisticRegression()
classifier.fit(X,Y)



classifier.coef_ # coefficients of features 
classifier.predict_proba (X) # Probability values 

y_pred = classifier.predict(X)
affair["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([affair,y_prob],axis=1)

######Confusion Matrix##########
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)
type(y_pred)
accuracy = sum(Y==y_pred)/affair.shape[0]
pd.crosstab(y_pred,Y)

##################################################################################
"""
Output variable -> y
y -> Whether the client has subscribed a term deposit or not 
Binomial ("yes" or "no")
"""

# Importing the libraries
import pandas as pd

dataset = pd.read_csv("C:\\DataScience\\Assignments\\Logistic Regression\\bank-full.csv", delimiter=";")

# Getting the barplot for the categorical columns 
dataset.shape

bank_data = dataset

#Creating Dummy columns for Categorical Variables

bank_data.head()


bank_data.loc[:,'default_im'] = bank_data.default.map(dict(yes=1,no=0)) 

bank_data.loc[:,"marital_im"] = pd.Series(map(lambda x: dict(single=1, married=2, divorced = 3)[x],
                                bank_data.marital.values.tolist()), bank_data.index)

bank_data.loc[:,'education_im'] = bank_data.education.map(dict(primary = 1, secondary= 2,tertiary=3,unknown=0)) 


bank_data = bank_data.drop(['default', 'marital','education'], axis = 1) 

bank_data.loc[:,'op'] = bank_data.y.map(dict(yes=1,no=0)) 
bank_data = bank_data.drop(['y'], axis = 1) 

bank_data.head()

# Model building 
from sklearn.linear_model import LogisticRegression

#leaving the cols 1,3,4,5,6,7

X = bank_data.iloc[:,[0,2,8,9,11,13,14,15]]
#Y = affair.iloc[:,0]               # Series Object created
Y = bank_data.iloc[:,16].values
classifier = LogisticRegression()
classifier.fit(X,Y)



classifier.coef_ # coefficients of features 
classifier.predict_proba (X) # Probability values 

y_pred = classifier.predict(X)
bank_data["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([bank_data,y_prob],axis=1)

######Confusion Matrix##########
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)

type(y_pred)
accuracy = sum(Y==y_pred)/bank_data.shape[0]
pd.crosstab(y_pred,Y)

############We have got anu accuracy of 88% with these data