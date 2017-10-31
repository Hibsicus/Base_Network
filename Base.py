# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import pickle

from sklearn import preprocessing #正規化數據

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
"""
用於輸出層的恆等函數和softmax函數
softmax的輸出總合為1，所以可以用來解釋機率
"""

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

a = np.array([0.3, 2.9, 4.4])
y = softmax(a)

print(y)
print(np.sum(y))


#%%
"""
MNIST的一些用法
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#訓練影像  訓練標籤
x_train, t_train = mnist.train.next_batch(1)
print(x_train.shape)
print(t_train.shape)

#測試影像  測試標籤
x_test, t_test = mnist.test.next_batch(1)
print(x_test.shape)
print(t_test.shape)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
    #img = preprocessing.scale(img) 正規化資料
    
    plt.gray()
    plt.imshow(img)
    plt.show()
    
img = x_train
label = t_train
print(label) 

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)

#%%
"""
神經網路的簡易模型
"""

def get_data(Normalize=True):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    x_train = mnist.train.images
    t_train = mnist.train.labels
    x_test = mnist.test.images
    t_test = mnist.test.labels
    
    if not Normalize:
        return  x_test, t_test
    else:
        x_test = preprocessing.scale(x_test)
        return x_test, t_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
        
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y


x, t = get_data(False)
network = init_network()

accury_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)

    if p == np.argmax(t[i]):
        accury_cnt += 1
        
print("Accuracy: " + str(float(accury_cnt) / len(x)))

"""
批次處理
"""

x, _t = get_data(False)
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(x_batch, axis=1)
    accuracy_cnt += np.sum(p == np.argmax(t[i:i+batch_size]))
    
print("Accuracy: " + str(float(accury_cnt) / len(x)))

#%%
"""
損失函數(loss function)
"""

#均方誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

#正確答案是2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))


#交叉熵誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

#批次對應交叉熵誤差
def cross_entropy_error_with_batch(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = t.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size


#%%
"""
微分...
"""

#數值微分
def numerical_diff(f, x):
    h = 1e-4 #0.0001
    return (f(x+h) - f(x-h)) / (2*h)

#example..
#y = 0.01x^2 + 0.1x
def function_1(x):
    return (0.01 * (x**2)) + (0.1*x)


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()

#%%
"""
梯度
"""

#example
#f(x0, x1) = x0^2 + x1^2
def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
            
        return grad

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()

grad = numerical_gradient(function_2, np.array([X, Y]))

plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.legend()
plt.draw()
plt.show()

#%%
"""
梯度下降
"""

def gradient_descent(f, init_x, lr = 0.01, setup_num=100):
    x = init_x
    
    for i in range(setup_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x

#example
init_x = np.array([-3.0, 4.0])
print(function_2(gradient_descent(function_2, init_x=init_x)))

#%%
"""
用簡易的神經網路實際計算梯度
"""

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x ,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
    
net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
 
print(np.argmax(p))

t = np.array([0, 0, 1])
print(net.loss(x, t))

def f(W):
    return net.loss(x, t)

dw = numerical_gradient(f, net.W)
print(dw)
#%%