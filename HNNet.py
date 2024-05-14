# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:00:25 2024

@author: Jinwang Feng; jinwangfeng11@163.com
"""

import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import time

##########################训练样本######################
file = 'F:/Feng_CNU_Work/Z-Datasets/Diabetes/Set1/train/T-01/*.jpg'
coll = io.ImageCollection(file)
X_train1 = np.asarray(coll)
y_train = np.hstack((np.ones(800),np.zeros(800)))
X_train = []
for i in range(X_train1.shape[0]):
    h1,_ = np.histogram(X_train1[i,:,:,0],256)
    h2,_ = np.histogram(X_train1[i,:,:,1],256)
    h3,_ = np.histogram(X_train1[i,:,:,2],256)
#    h = 0.299*h1[1:254]+0.587*h2[1:254]+0.114*h3[1:254]
    h1 = h1[1:231]
    h1 = 0.299*h1/h1.max()
    h2 = h2[1:231]
    h2 = 0.587*h2/h2.max()
    h3 = h3[1:231]
    h3 = 0.114*h3/h3.max()
    h = np.hstack((h1,h2,h3))   
    X_train.append(h)
del X_train1, h1, h2, h3, h
X_train = np.asarray(X_train)
X_train = X_train.T

##########################验证样本######################
file = 'F:/Feng_CNU_Work/Z-Datasets/Diabetes/Set1/val/T-01/*.jpg'
coll = io.ImageCollection(file)
X_test1 = np.asarray(coll)
y_val = np.hstack((np.ones(100),np.zeros(100)))
X_val = []
for i in range(X_test1.shape[0]):
    h1,_ = np.histogram(X_test1[i,:,:,0],256)
    h2,_ = np.histogram(X_test1[i,:,:,1],256)
    h3,_ = np.histogram(X_test1[i,:,:,2],256)    
    h1 = h1[1:231]
    h1 = 0.299*h1/h1.max()
    h2 = h2[1:231]
    h2 = 0.587*h2/h2.max()
    h3 = h3[1:231]
    h3 = 0.114*h3/h3.max()
    h = np.hstack((h1,h2,h3))    
    X_val.append(h)
del X_test1, h1, h2, h3, h
X_val = np.asarray(X_val)
X_val = X_val.T


##########################测试样本######################
file = 'F:/Feng_CNU_Work/Z-Datasets/Diabetes/Set1/test/T-01/*.jpg'
coll = io.ImageCollection(file)
X_test1 = np.asarray(coll)
y_test = np.hstack((np.ones(100),np.zeros(100)))
X_test = []
for i in range(X_test1.shape[0]):
    h1,_ = np.histogram(X_test1[i,:,:,0],256)
    h2,_ = np.histogram(X_test1[i,:,:,1],256)
    h3,_ = np.histogram(X_test1[i,:,:,2],256)    
    h1 = h1[1:231]
    h1 = 0.299*h1/h1.max()
    h2 = h2[1:231]
    h2 = 0.587*h2/h2.max()
    h3 = h3[1:231]
    h3 = 0.114*h3/h3.max()
    h = np.hstack((h1,h2,h3))    
    X_test.append(h)
del X_test1, h1, h2, h3, h
X_test = np.asarray(X_test)
X_test = X_test.T


##########################DNN模型参数初始化######################
def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

##########################正向传播######################
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A
def relu(Z):
    A = np.maximum(-0.01, Z)
    return A
def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    cache = (A_prev, W, b, Z)
    return A, cache
def model_forward(X, parameters):
    caches = []
    A = X
    L = int(len(parameters)/2)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches
def compute_cost(AL, Y):
    m = AL.shape[1]
    cost = -1/m*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)
    return cost

##########################反向传播######################
def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= -0.01] = -0.01
    return dZ
def sigmoid_backward(dA, Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA*s*(1-s)
    return dZ
def linear_activation_backward(dA, cache, activation):
    A_prev, W, b, Z = cache
    if activation == "relu":
        dZ = relu_backward(dA, Z)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    m = dA.shape[1]
    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db
def model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

##########################模型参数更新######################
def update_parameters(parameters, grads, learning_rate):
    L = int(len(parameters)/2)
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate*grads["db" + str(l+1)]
    return parameters

##########################DNN模型搭建######################
def nn_model(X, Y, layers_dims, learning_rate=0.01, num_iterations=3000, print_cost=False):
#    np.random.seed(1)
    costs = []
    parameters = initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %d: %f" %(i, cost))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (Per hundred)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show
    return parameters

##########################DNN模型预测######################
def predict(X, y, parameters):
    AL, caches = model_forward(X, parameters)
    predictions = AL > 0.5
    acc = np.mean(predictions==y)*100
    print("Accuracy: %f%%" % acc)
    return predictions

#####################################################################################
learning_rate = 0
num_iterations = 1001
layers_dims = [690, 128, 10, 1]

##########################2-DNN模型Training-Validation-Testing######################
time1 = time.time()
parameters = nn_model(X_train, y_train, layers_dims, learning_rate, num_iterations, print_cost=True)
time2 = time.time()
Train_time = time2 - time1

print('Training：')
pred_train = predict(X_train, y_train, parameters)
print(Train_time)

print('Validation：')
pred_test = predict(X_val, y_val, parameters)


print('Test：')
time3 = time.time()
pred_test = predict(X_test, y_test, parameters)
time4 = time.time()
pred_time = time4 - time3
print(pred_time)


