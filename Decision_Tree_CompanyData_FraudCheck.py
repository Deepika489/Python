# -*- coding: utf-8 -*-
"""
About the data: 
Let’s consider a Company dataset with around 10 variables and 400 records. 
The attributes are as follows: 
 Sales -- Unit sales (in thousands) at each location
 Competitor Price -- Price charged by competitor at each location
 Income -- Community income level (in thousands of dollars)
 Advertising -- Local advertising budget for company at each location (in thousands of dollars)
 Population -- Population size in region (in thousands)
 Price -- Price company charges for car seats at each site
 Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
 Age -- Average age of the local population
 Education -- Education level at each location
 Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
 US -- A factor with levels No and Yes to indicate whether the store is in the US or not
The company dataset looks like this: 
 
Problem Statement:
A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  

"""

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing dataset
company_ds = pd.read_csv("C:\\DataScience\\Assignments\\Decision Tree\\Company_Data.csv")



#Creating Dummy columns for Categorical Variables

company_ds.loc[:,'ShelveLocnum'] = company_ds.ShelveLoc.map(dict(Bad = 0,Medium=1,Good = 2))
company_ds.loc[:,'Urbannum'] = company_ds.Urban.map(dict(Yes=1,No = 0))
company_ds.loc[:,'USnum'] = company_ds.US.map(dict(Yes=1,No = 0))

print (company_ds.dtypes)

data = company_ds.iloc[:,[0,1,2,3,4,5,7,8,11,12,13]]

data.loc[ data.Sales < 5.0, 'SO' ] = 'small'
data.loc[(data.Sales >= 5.0 ) & (data.Sales < 10.0), 'SO'] = 'medium'
data.loc[ data.Sales >= 5.0, 'SO' ] = 'large'

ip = data.iloc[:,1:11]
op = data.iloc[:,11]

# Splitting data into training and testing data set

from sklearn.model_selection import train_test_split
ip_train,ip_test,op_train,op_test = train_test_split(ip,op, test_size = 0.2, random_state= 0)


#implementing Decision Tree Classifier
from sklearn.tree import  DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier = classifier.fit(ip_train,op_train)

preds = classifier.predict(ip_test)

pd.crosstab(op_test,preds)

np.mean(preds==op_test) # 0.8


"""
Use decision trees to prepare a model on fraud data 
treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

Data Description :

Undergrad : person is under graduated or not
Marital.Status : marital status of a person
Taxable.Income : Taxable income is the amount of how much tax an individual owes to the government 
Work Experience : Work experience of an individual person
Urban : Whether that person belongs to urban area or not
"""


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing dataset
Fraud_ds = pd.read_csv("C:\\DataScience\\Assignments\\Decision Tree\\Fraud_check.csv")

Fraud_ds.rename(columns = {'Marital.Status': 'Marital_Stat' ,'Taxable.Income': 'Tax_Inc','City.Population' : 'City_Pop', 'Work.Experience' :'Work_Exp' }, inplace = True)

colnames = list(Fraud_ds.columns)
#Creating Dummy columns for Categorical Variables

Fraud_ds.loc[:,'Undergradnum'] = Fraud_ds.Undergrad.map(dict(YES=1,NO = 0))
Fraud_ds.loc[:,'MStatusnum'] = Fraud_ds.Marital_Stat.map(dict(Single=1,Married = 2, Divorced = 0))
Fraud_ds.loc[:,'Urbannum'] = Fraud_ds.Urban.map(dict(YES=1,NO = 0))

#Creating Categorical op variable
Fraud_ds.loc[ Fraud_ds.Tax_Inc  <= 30000, 'SO' ] = 'Risky'
Fraud_ds.loc[ Fraud_ds.Tax_Inc  > 30000, 'SO' ] = 'Good'

ip = Fraud_ds.iloc[:,[2,3,4,6,7,8]]
op = Fraud_ds.iloc[:,9]

# Splitting data into training and testing data set

from sklearn.model_selection import train_test_split
ip_train,ip_test,op_train,op_test = train_test_split(ip,op, test_size = 0.2, random_state= 0)


#implementing Decision Tree Classifier
from sklearn.tree import  DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier = classifier.fit(ip_train,op_train)

preds = classifier.predict(ip_test)

pd.crosstab(op_test,preds)

np.mean(preds==op_test) # 100
