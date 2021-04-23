#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy.stats import multivariate_normal
#%%
#reading dataset
dataset = pd.read_csv('test-Minimum-Error-Rate-Classifier.txt', header= None)
X_test = dataset.iloc[:,:].values
#%%
#declaring constants
miu_1 = np.array([0,0])
miu_2 = np.array([2,2])

sigma_1 = np.array([[0.25,0.3],
                    [0.3,1]])
sigma_2 = np.array([[0.5,0],
                    [0,0.5]])

prior_1 = 0.5
prior_2 = 0.5

class_1 = []
class_2 = []
#%%
#predicting class
for x in range(len(X_test)):
    #class 1
    pst_1 = prior_1* 1/math.sqrt((2*math.pi)**2 * np.linalg.det(sigma_1)) * math.exp(-0.5*np.matmul((X_test[x].reshape(1,2)-miu_1),np.matmul(np.linalg.inv(sigma_1), np.transpose(X_test[x].reshape(1,2)-miu_1))))
    pst_2 = prior_2* 1/math.sqrt((2*math.pi)**2 * np.linalg.det(sigma_2)) * math.exp(-0.5*np.matmul((X_test[x].reshape(1,2)-miu_2),np.matmul(np.linalg.inv(sigma_2), np.transpose(X_test[x].reshape(1,2)-miu_2))))
    if(pst_1>pst_2):
        class_1.append(X_test[x])
    else:
        class_2.append(X_test[x])
#%%
#ploting values
X_class_1 = np.array(class_1)
X_class_2 = np.array(class_2)        

plt.scatter(X_class_1[:,0],X_class_1[:,1], color ='green', label='Test Class 1')
plt.scatter(X_class_2[:,0],X_class_2[:,1], color ='red', label='Test Class 2')

plt.legend(loc='upper left', fontsize=7)
plt.axes().set_xlabel('X1')
plt.axes().set_ylabel('X2')
plt.show()
#%%
#Plotting contour graph

fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.linspace(-10,10,500)
y = np.linspace(-10,10,500)
X, Y = np.meshgrid( x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X 
pos[:, :, 1] = Y

mean_1 = np.array([0,0])
cov_1  = sigma_1
rv_1 = multivariate_normal(mean_1,cov_1)

mean_2 = np.array([2,2])
cov_2  = sigma_2
rv_2 = multivariate_normal(mean_2,cov_2)

# Plot the 3D surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv_1.pdf(pos), rstride=8, cstride=8, alpha=0.3)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv_2.pdf(pos), rstride=8, cstride=8, alpha=0.3)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

cset = ax.contour(X, Y, rv_1.pdf(pos), zdir='z', offset=-0.4, cmap=cm.coolwarm)
cset = ax.contour(X, Y, rv_2.pdf(pos), zdir='z', offset=-0.4, cmap=cm.coolwarm)

ax.scatter(X_class_1[:,0], X_class_1[:,1], 0, marker='o',color='green')
ax.scatter(X_class_2[:,0], X_class_2[:,1], 0, marker='o',color='red')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim([-6,6])
ax.set_ylim([-6,6])
ax.set_zlim([-.5,.5])

#Decision Boundary
W = np.linalg.inv(sigma_1) - np.linalg.inv(sigma_2)
w = np.matmul(miu_1, np.linalg.inv(sigma_1)) - np.matmul(miu_2, np.linalg.inv(sigma_2))
w = w*2


w0 = - np.log(np.linalg.det(sigma_2)/np.linalg.det(sigma_1)) + 2*np.log(prior_1/prior_2) + np.matmul(np.matmul(miu_1, np.linalg.inv(sigma_1)), miu_1) - np.matmul(np.matmul(miu_2, np.linalg.inv(sigma_2)), miu_2)

db = []
i = 0.0
for j in range(-80,25):
  i = j*0.1
  c = W[0][0]*i*i - w[0]*i + w0
  b = W[0][1]*i + W[1][0]*i - w[1]
  a = W[1][1]
  
  y = (-b+np.sqrt(b*b - 4*a*c))/(2*a)
  db.append([i, y])

plt.plot([db[i][0] for i in range(len(db))], [db[i][1] for i in range(len(db))])

plt.show()  

#%%

