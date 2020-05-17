# -*- coding: utf-8 -*-
"""
@author: Deepika.J
Desc : Perform Clustering for the crime data and identify the number of clusters formed and draw inferences.

Data Description:
Murder -- Muder rates in different places of United States
Assualt- Assualt rate in different places of United States
UrbanPop - urban population in different places of United States
Rape - Rape rate in different places of United States

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing crime dataset
crime_data = pd.read_csv("C:\\DataScience\\Assignments\\Clustering\\crime_data.csv")

#checking the crime for murder, assault and Rape  
# As all the crime data are in same normalised form we dont need to perform any EDA functions.
X = crime_data.iloc[:,[1,2,4]].values

#using Dendrogram to find the optimal no of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'single', metric = 'euclidean'))
plt.xlabel("Crime_Rate");plt.ylabel("Eucledian Distances")
plt.show()

# We can create Linkage data and then draw Dendrogram using best Linkage functions
from scipy.cluster.hierarchy import linkage 
z = linkage(X, method="complete",metric="euclidean")


plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Crime Data');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


D = linkage(X, 'ward')
fig = plt.figure(figsize=(25, 10))
dn = sch.dendrogram(D)
    
# From the Inferences we are going for Complete Linkage function as better option.
#Fitting Hierarchial Clustering to dataset 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity ='euclidean', linkage = 'complete')
y = hc.fit(X)

y_hc = hc.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'yellow', label = "Careful Zone (Medium Crimes)")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'Red', label = "Danger Zone (Higher crimes)")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = "Normal Zone (Less Crimes)")
plt.legend()
plt.show()

#Adding the Clusters to the Original Crime Report
hc.labels_
cluster_labels=pd.Series(hc.labels_)

crime_data['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime_data = crime_data.iloc[:,[5,0,1,2,3,4]]
crime_data.head()

# getting aggregate mean of each cluster
crime_data.groupby(crime_data.clust).mean()

# creating a csv file 
crime_data.to_csv("CrimeDate_WC.csv",index=False) #,encoding="utf-8")

##############           K _ Means             ##################

# Importing the dataset
crime_data = pd.read_csv("C:\\DataScience\\Assignments\\Clustering\\crime_data.csv")
X = crime_data.iloc[:,[1,2,4]].values
# y = dataset.iloc[:, 3].values

# Using the elbow method of KMeans to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

###### screw plot or elbow curve ############
from scipy.spatial.distance import cdist 
k = list(range(2,15))

X_df = pd.DataFrame(data = X)

TWSS = [] # variable for storing total within sum of squares for each kmeans 

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(X_df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,X_df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4) 
model.fit(X)


# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Crime Data Analysis')
plt.xlabel('Crime Rate')
plt.ylabel('Rate Count')
plt.legend()
plt.show()