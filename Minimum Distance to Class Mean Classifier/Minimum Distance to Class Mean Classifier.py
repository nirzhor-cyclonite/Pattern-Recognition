#--------------------------------Importing Libraries-----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
#--------------------------------Reading Dataset-----------------------------

dataset_train = pd.read_csv('train.txt', sep=' ', header= None)
class_1 = dataset_train[dataset_train.iloc[:,2]==1]
class_2 =  dataset_train[dataset_train.iloc[:,2]==2]
X_train_1 = class_1.iloc[:,:-1].values
X_train_2 = class_2.iloc[:,:-1].values

dataset_test = pd.read_csv('test.txt', sep=' ', header = None)
X_test = dataset_test.iloc[:,:-1].values
Y_test = dataset_test.iloc[:,2].values

#%%
#--------------------------------Calculating g(x)-----------------------------

miu_1 = np.mean(X_train_1, axis = 0)
miu_1 = miu_1.reshape(2,1)
miu_2 = np.mean(X_train_2, axis = 0)
miu_2 = miu_2.reshape(2,1)
g1_x = []
g2_x = []

x_lst_1 =[]
x_lst_2 =[]

for x in range(len(X_test)):
    g1_x.append(np.matmul(np.transpose(X_test[x]),miu_1) - (1/2)* np.matmul(np.transpose(miu_1), miu_1))
    g2_x.append(np.matmul(np.transpose(X_test[x]),miu_2) - (1/2)* np.matmul(np.transpose(miu_2), miu_2))
    
#%%
#--------------------------------Predicting Class-----------------------------
    
y_lst = []
for x in range(len(X_test)):
    if(g1_x[x]>g2_x[x]):
        x_lst_1.append(X_test[x])
        y_lst.append(1)
    else:
        x_lst_2.append(X_test[x])
        y_lst.append(2)
        
X_test_1 = np.array(x_lst_1)
X_test_2 = np.array(x_lst_2)
y_pred = np.array(y_lst)
#%%
#--------------------------------Plotting Results-----------------------------

plt.scatter(X_train_1[:,0],X_train_1[:,1], color ='green', label='Train Class 1')
plt.scatter(X_train_2[:,0],X_train_2[:,1], color ='red', label='Train Class 2')
plt.scatter(X_test_1[:,0], X_test_1[:,1], color = 'green', marker = '*', label='Test Class 1')
plt.scatter(X_test_2[:,0], X_test_2[:,1], color = 'red', marker = '*', label='Test Class 2')
plt.scatter(miu_1[0], miu_1[1], color = 'green', marker = 's', label = 'Class Mean 1')
plt.scatter(miu_2[0], miu_2[1], color = 'red', marker = 's', label = 'Class Mean 2')

line_x = []
line_y =[]
temp = 0.5 * (np.matmul(np.transpose(miu_1),miu_1) - np.matmul(np.transpose(miu_2),miu_2))
x = -4
while(x<4):
    line_x.append(x)
    line_y.append(((temp - x*(miu_1[0]-miu_2[0]))/(miu_1[1]-miu_2[1])).item())
    x = x + 0.1

plt.plot(np.array(line_x), np.array(line_y), '-b')
plt.legend(loc='upper right', fontsize=7)
plt.show()
#%%
#--------------------------------Calculating Accuracy-----------------------------

cnt = 0
for i in range(len(X_test)):
    if Y_test[i] == y_pred[i]:
        cnt = cnt+1

print('Accuracy =', cnt/X_test.shape[0])
