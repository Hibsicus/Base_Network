# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt


"""
    感知器

    y = b + (w * x)

"""

#%%
"""
活化函數
"""

#階梯函數
def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

#Sigmoid函數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

#ReLU函數
def relu(x):
    return np.maximum(0, x)

y = relu(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

#%%
"""
例子
非常簡易的三層神經網路
"""
#輸入層
X = np.array([1.0, 0.5])

W1 = np.array([[0.1, 0.3, 0.5],
               [0.2, 0.4, 0.6]])
    
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1


#隱藏層
Z1 = sigmoid(A1)

print(A1)
print(Z1)


#輸出層
W2 = np.array([[0.1, 0.3], [0.2, 0.4], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2

print(A2.shape)
print(A2)
Y = A2
#%%