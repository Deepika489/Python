# -*- coding: utf-8 -*-
"""
Data Description:
 
The file EastWestAirlinescontains information on passengers who belong to an airline’s 
frequent flier program. For each passenger the data include information on their 
mileage history and on different ways they accrued or spent miles in the last year. 
The goal is to try to identify clusters of passengers that have similar characteristics 
for the purpose of targeting different segments for different types of mileage offers

ID --Unique ID
Balance--Number of miles eligible for award travel
Qual_mile--Number of miles counted as qualifying for Topflight status
cc1_miles?	CHAR--Has member earned miles with airline freq. flyer credit card in the past 12 months (1=Yes/0=No)?
cc2_miles?	CHAR--Has member earned miles with Rewards credit card in the past 12 months (1=Yes/0=No)?
cc3_miles?	--Has member earned miles with Small Business credit card in the past 12 months (1=Yes/0=No)?
Bonus_miles--Number of miles earned from non-flight bonus transactions in the past 12 months
Bonus_trans--Number of non-flight bonus transactions in the past 12 months
Flight_miles_12mo--Number of flight miles in the past 12 months
Flight_trans_12--Number of flight transactions in the past 12 months
Days_since_enrolled--Number of days since enrolled in flier program
Award--whether that person had award flight (free flight) or not
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Impotrting the Airlines Data
Air = pd.read_csv("C:\\DataScience\\Assignments\\Clustering\\EastWestAirlines.csv")

# To get the different segment users we are going to use the columns for Clustering.
## cc1_miles? cc2_miles? cc3_miles?, Bonus_Trans, Flight_Trans_12
## Based on this we will prepare the inferences

data_ip = Air.iloc[:,[1,3,4,5,7,9]]

##Normalising the Data
def norm_fun(i):
   x = ( i - i.min() / i.max() - i.min() )
   return(x)    

df_norm = norm_fun(data_ip)

######## Hierarchial Clustering ##########

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage (data_ip, method = 'complete',  metric = 'euclidean')
sch.dendrogram(z, leaf_rotation = 0., leaf_font_size = 8., )

sch.dendrogram(z)

fig = plt.figure(figsize=(25, 10))
dn = sch.dendrogram(z)

#Fitting Hierarchial Clustering to dataset 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity ='euclidean', linkage = 'complete')
y_hc = hc.fit_predict(data_ip)

#Adding the Clusters to the Original Crime Report
hc.labels_
cluster_labels=pd.Series(hc.labels_)

Air['clust']=cluster_labels # creating a  new column and assigning it to new column 
Air = Air.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
Air.head()

# getting aggregate mean of each cluster
Air.groupby(Air.clust).mean()

data_ar = np.array(data_ip)
#Visualizing the clusters
plt.scatter(data_ar[y_hc == 0, 0], data_ar[y_hc == 0, 1], s = 50, c = 'yellow', label = "Perfect Customers with more Award")
plt.scatter(data_ar[y_hc == 1, 0], data_ar[y_hc == 1, 1], s = 50, c = 'Pink', label = "Very low suage of flight Customers")
plt.scatter(data_ar[y_hc == 2, 0], data_ar[y_hc == 2, 1], s = 50, c = 'Red', label = "Good Pick for Offers")
plt.scatter(data_ar[y_hc == 3, 0], data_ar[y_hc == 3, 1], s = 50, c = 'blue', label = "Already in hight rate of travelling and less need of awards")
plt.scatter(data_ar[y_hc == 4, 0], data_ar[y_hc == 4, 1], s = 50, c = 'green', label = "Beeter Pick for offers")
plt.legend()
plt.show()

######## KMeans Clustering     ##########
# Using the elbow method of KMeans to find the optimal number of clusters

###### screw plot or elbow curve ############
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
k = list(range(2,15))

TWSS = [] # variable for storing total within sum of squares for each kmeans 

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data_ip)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(data_ip.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,data_ip.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(data_ip)



model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
Air['clust']=md
Air.head(10) # creating a  new column and assigning it to new column 
Air = Air.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,0]]
Air.iloc[:,1:13].groupby(Air.clust).mean()
