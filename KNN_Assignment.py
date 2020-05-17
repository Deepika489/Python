# -*- coding: utf-8 -*-
"""
Prepare a model for glass classification using KNN
Data Description:
RI : refractive index
Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
Mg: Magnesium
AI: Aluminum
Si: Silicon
K:Potassium
Ca: Calcium
Ba: Barium
Fe: Iron

Type: Type of glass: (class attribute)
1 -- building_windows_float_processed
 2 --building_windows_non_float_processed
 3 --vehicle_windows_float_processed
 4 --vehicle_windows_non_float_processed (none in this database)
 5 --containers
 6 --tableware
 7 --headlamps

"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # library to do visualizations 

glass_data = pd.read_csv("C:\\DataScience\\Assignments\\KNN\\glass.csv")

X = glass_data.iloc[:,[0,1,2,3,4,5,6,7,8]]
Y = glass_data.iloc[:,9].values

# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
classifier = KNC(n_neighbors = 3)
classifier.fit(X,Y)

#  accuracy 
glass_data_acc = np.mean(classifier.predict(X)==Y) # 84 %


# for 5 nearest neighbours 
classifier = KNC(n_neighbors = 5)
classifier.fit(X,Y)

#  accuracy 
glass_data_acc = np.mean(classifier.predict(X)==Y) # 76.16 %
 
# creating empty list variable 
acc = []

for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(X,Y)
    data_acc = np.mean(neigh.predict(X)==Y)
    acc.append(data_acc)

#############neighbours with 3 has given the highest accuracy so we can keep that as the best model.



""""
Implement a KNN model to classify the animals in to categorie
""""

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # library to do visualizations 

Zoo_data = pd.read_csv("C:\\DataScience\\Assignments\\KNN\\Zoo.csv")

X = Zoo_data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
Y = Zoo_data.iloc[:,17].values

# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
classifier = KNC(n_neighbors = 3)
classifier.fit(X,Y)

#  accuracy 
Zoo_data_acc = np.mean(classifier.predict(X)==Y) # 98 %


# for 5 nearest neighbours 
classifier = KNC(n_neighbors = 5)
classifier.fit(X,Y)

#  accuracy 
Zoo_data_acc = np.mean(classifier.predict(X)==Y) # 94 %
 
# creating empty list variable 
acc = []

for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(X,Y)
    data_acc = np.mean(neigh.predict(X)==Y)
    acc.append(data_acc)

# the K value 3 has given the highest accuracy.