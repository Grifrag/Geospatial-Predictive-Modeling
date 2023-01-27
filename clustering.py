import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv


df_in = pd.read_csv('Cruise_Haugesund_Skudefjorden_In_20210212.csv')
df_out=pd.read_csv("Cruise_Haugesund_Skudefjorden_Out_20210212.csv")

df=df_in.append(df_out,ignore_index=True)
print(df)



#New dataframe without name
New=df.loc[:,['id','lat','lon']]

K_clusters = range(1,10)

kmeans = [KMeans(n_clusters=i) for i in K_clusters]

Y_axis = New[['lat']]
X_axis = New[['lon']]

score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()





kmeans = KMeans(n_clusters = 3, init ='k-means++')
kmeans.fit(New[New.columns[1:3]]) # Compute k-means clustering. # Compute k-means clustering.

New['cluster_label'] = kmeans.fit_predict(New[New.columns[1:3]])

centers = kmeans.cluster_centers_ # Coordinates of cluster centers.

labels = kmeans.predict(New[New.columns[1:3]]) # Labels of each point

print(New)



New.plot.scatter(x = 'lat', y = 'lon', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()



