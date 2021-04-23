import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
#%%
#reading dataset
dataset = pd.read_csv('data_k_mean.txt', sep=' ', header = None)
plot_points = dataset.values
plt.scatter(plot_points[:,0],plot_points[:,1], color ='green', label='Data Points')
plt.axes().set_xlabel('X1')
plt.axes().set_ylabel('X2')
plt.show()
#%%
centroid_1 = dataset.iloc[random.randint(0,3000-1),:].values
centroid_2 = dataset.iloc[random.randint(0,3000-1),:].values


change = True
cluster_1 = []
cluster_2 = []
while(change):
    change = False
    
    cluster_1 = []
    cluster_2 = []
    cluster_1.append(centroid_1)
    cluster_2.append(centroid_2)
    
    previous_centroid_1 = centroid_1
    previous_centroid_2 = centroid_2
    
    for x in range(len(plot_points)):
        distance_1 = math.sqrt((plot_points[x][0]-centroid_1[0])**2 + (plot_points[x][1]-centroid_1[1])**2)
        distance_2 = math.sqrt((plot_points[x][0]-centroid_2[0])**2 + (plot_points[x][1]-centroid_2[1])**2)
        if(distance_1<distance_2):
            cluster_1.append(plot_points[x])
        else:
            cluster_2.append(plot_points[x])
            
    centroid_1 = np.mean(np.array(cluster_1),axis = 0)
    centroid_2 = np.mean(np.array(cluster_2),axis = 0)
    
    
    if(np.array_equal(centroid_1, previous_centroid_1) and np.array_equal(centroid_2, previous_centroid_2)):
        change = False
    else:
        change = True
#%%
cluster_1 = np.array(cluster_1)
cluster_2 = np.array(cluster_2)

plt.scatter(cluster_1[:,0],cluster_1[:,1], color ='blue', label='Cluster 1')
plt.scatter(cluster_2[:,0],cluster_2[:,1], color ='red', label='Cluster 2')
plt.axes().set_xlabel('X1')
plt.axes().set_ylabel('X2')
plt.show()
#%%