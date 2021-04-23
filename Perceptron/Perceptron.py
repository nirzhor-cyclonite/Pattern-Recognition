#--------------------------------Importing Libraries-----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
random.seed(10)
#%%
#-----------------------------Task 1---------------------------------------------
dataset = pd.read_csv('train-perceptron.txt', sep=' ', header= None)
class_1 = dataset[dataset.iloc[:,2]==1]
class_2 =  dataset[dataset.iloc[:,2]==2]

X_plt_1 =  class_1.iloc[:,:-1].values
X_plt_2 = X_train_2 = class_2.iloc[:,:-1].values

plt.scatter(X_plt_1[:,0],X_plt_1[:,1], color ='green', label='Train Class 1')
plt.scatter(X_plt_2[:,0],X_plt_2[:,1], color ='red', label='Train Class 2')

plt.legend(loc='upper left', fontsize=7)
plt.axes().set_xlabel('X1')
plt.axes().set_ylabel('X2')
plt.show()

#%%
#-----------------------------Task 2---------------------------------------------
higher_data = pd.DataFrame({'x1^2':dataset.iloc[:,0]*dataset.iloc[:,0], 'x2^2':dataset.iloc[:,1]*dataset.iloc[:,1],
                           'x1*x2':dataset.iloc[:,0]*dataset.iloc[:,1], 'x1': dataset.iloc[:,0], 
                           'x2':dataset.iloc[:,1], '1':1, 'label': dataset.iloc[:,2] })

#normalization
y = higher_data.apply( lambda x:x*-1 if x.iloc[6]==2 else x, axis= 1).iloc[:,:-1].values
y[[2,3,4,5]] = y[[5,2,3,4]]

#%%
#-----------------------------Task 3 and 4---------------------------------------
np.random.seed(10)
weight = [np.ones(shape = (1,6)), np.zeros(shape = (1,6)), np.random.rand(1,6)]

num_of_data = len(dataset.index)

weight_zero_iterations =[]
weight_one_iterations = []
weight_rand_iterations = []

for x in range (3):
    temp_list_one = []
    temp_list_many = []
    for alp in range (10):
        alpha = (alp+1)*0.1
        
        #many at a time
        w = weight[x]
        proper_classified = 0
        itr = 0
        while(proper_classified != num_of_data ):
            itr = itr + 1
            proper_classified = 0
            y_sum = np.zeros(shape = (1,6))
            for i in range (num_of_data):
                wTy = np.matmul(w, y[i,:].reshape(6,1))
                if(wTy>0):
                    proper_classified =  proper_classified + 1
                else:
                    y_sum = y_sum + y[i,:].reshape(1,6)
    
            w = w + alpha*y_sum
        temp_list_many.append(itr)
        
        #many at a time
        w = weight[x]
        proper_classified = 0
        itr = 0
        while(proper_classified != num_of_data ):
            itr = itr + 1
            proper_classified = 0
            y_sum = np.zeros(shape = (1,6))
            for i in range (num_of_data):
                wTy = np.matmul(w, y[i,:].reshape(6,1))
                if(wTy>0):
                    proper_classified =  proper_classified + 1
                else:
                    w = w + alpha*y[i,:].reshape(1,6)

        temp_list_one.append(itr)
        
    if(x==1):
        weight_zero_iterations.append(temp_list_one)
        weight_zero_iterations.append(temp_list_many)
    elif (x==0):
        weight_one_iterations.append(temp_list_one)
        weight_one_iterations.append(temp_list_many)
    else:
        weight_rand_iterations.append(temp_list_one)
        weight_rand_iterations.append(temp_list_many)

#%%
#----------------------------Ploting Bar Chart-----------------------------------
#all one
alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
one_at = weight_one_iterations[0]
many_at = weight_one_iterations[1]

x = np.arange(len(alpha))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, one_at, width, label='One at a time')
rects2 = ax.bar(x + width/2, many_at, width, label='Many at a time')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Learning Rate')
ax.set_ylabel('No of Iterations')
ax.set_title('Perceptron Comparison')
ax.set_xticks(x)
ax.set_xticklabels(alpha)
ax.legend()


fig.tight_layout()

plt.show()
#%%
#all zero
alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
one_at = weight_zero_iterations[0]
many_at = weight_zero_iterations[1]

x = np.arange(len(alpha))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, one_at, width, label='One at a time')
rects2 = ax.bar(x + width/2, many_at, width, label='Many at a time')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Learning Rate')
ax.set_ylabel('No of Iterations')
ax.set_title('Perceptron Comparison')
ax.set_xticks(x)
ax.set_xticklabels(alpha)
ax.legend()


fig.tight_layout()

plt.show()
#%%
#Random
alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
one_at = weight_rand_iterations[0]
many_at = weight_rand_iterations[1]

x = np.arange(len(alpha))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, one_at, width, label='One at a time')
rects2 = ax.bar(x + width/2, many_at, width, label='Many at a time')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Learning Rate')
ax.set_ylabel('No of Iterations')
ax.set_title('Perceptron Comparison')
ax.set_xticks(x)
ax.set_xticklabels(alpha)
ax.legend()


fig.tight_layout()

plt.show()
#%%