#--------------------------------Importing Libraries-----------------------------
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
#%%
#--------------------------------Reading Dataset-----------------------------

dataset_train = pd.read_csv('train_knn.txt', sep=',', header= None)
class_1 = dataset_train[dataset_train.iloc[:,2]==1]
class_2 =  dataset_train[dataset_train.iloc[:,2]==2]
X_train_1 = class_1.iloc[:,:-1].values
X_train_2 = class_2.iloc[:,:-1].values

dataset_test = pd.read_csv('test_knn.txt', sep=',', header = None)
X_test = dataset_test.iloc[:,:].values
#%%
#--------------------------------Designing Datapoint Class-----------------------------
class DataPoint():
    def __init__(self, _data_, _class_, _distance_):
        self._data_ = _data_
        self._class_ = _class_
        self._distance_ = _distance_

def sorter(e):
    return e._distance_
#%%
#--------------------------------KNN implementation-----------------------------
x_lst_1 =[]
x_lst_2 =[]
num_neighbors = int(input('Enter number of Neighbors: '))
file1 = open("prediction.txt","w+")
        
for x in range(len(X_test)):
    dist_list = []
    class_1_count = 0
    class_2_count = 0
    for y in range(len(X_train_1)):
        dist_list.append(DataPoint(X_train_1[y], 1, math.sqrt((X_test[x][0]-X_train_1[y][0])**2 + (X_test[x][1]-X_train_1[y][1])**2)))
    for y in range(len(X_train_2)):
        dist_list.append(DataPoint(X_train_2[y], 2, math.sqrt((X_test[x][0]-X_train_2[y][0])**2 + (X_test[x][1]-X_train_2[y][1])**2)))
        
    dist_list.sort(key=sorter)
    
    print('Test Point: ', X_test[x],file = file1)
    for i in range(num_neighbors):
        if(dist_list[i]._class_ == 1):
            class_1_count = class_1_count + 1
            print('Distance', i+1,':', dist_list[i]._distance_, '\tClass:', 1, file = file1)
        else:
            class_2_count = class_2_count + 1
            print('Distance', i+1,':', dist_list[i]._distance_, '\tClass:', 2,file = file1)
    
    if(class_1_count>class_2_count):
        x_lst_1.append(X_test[x])
        print('Predicted Class:',1, file = file1)
    else:
        x_lst_2.append(X_test[x])
        print('Predicted Class:',2, file = file1)
    #print(X_test[x])
file1.close()
print("Results written on file")
#%%
#--------------------------------Plotting Results-----------------------------
X_test_1 = np.array(x_lst_1)
X_test_2 = np.array(x_lst_2)

plt.scatter(X_train_1[:,0],X_train_1[:,1], color ='green', label='Train Class 1')
plt.scatter(X_train_2[:,0],X_train_2[:,1], color ='red', label='Train Class 2')
plt.scatter(X_test_1[:,0], X_test_1[:,1], color = 'green', marker = '*', label='Test Class 1')
plt.scatter(X_test_2[:,0], X_test_2[:,1], color = 'red', marker = '*', label='Test Class 2')


plt.legend(loc='upper left', fontsize=7)
plt.axes().set_xlabel('X1')
plt.axes().set_ylabel('X2')
plt.show()
#%%