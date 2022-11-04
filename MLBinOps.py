#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 10:55:47 2022

@author: monty
"""

import numpy as np

def init_params(layers):
    W1=np.random.rand(layers[1],layers[0])-0.5
    b1=np.random.rand(layers[1],1)-0.5
    W2=np.random.rand(layers[2],layers[1])-0.5
    b2=np.random.rand(layers[2],1)-0.5
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1=W1@X+b1
    A1=ReLU(Z1)
    Z2=W2@A1+b2
    A2=sigmoid(Z2)
    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    dZ2=A2-Y
    dW2=1/ln*(dZ2@A1.T)
    db2=1/ln*np.sum(dZ2,1)
    dZ1=(W2.T@dZ2)*deriv_ReLU(Z1)
    dW1=1/ln*dZ1.dot(X.T)
    db1=1/ln*np.sum(dZ1,1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, layers):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (layers[1],1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (layers[2],1))
    return W1, b1, W2, b2

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def ReLU(Z):
    return np.maximum(Z,0)

def deriv_ReLU(Z):
    return Z > 0

def deriv_sigmoid(Z):
    return sigmoid(Z)*(1-sigmoid(Z))

def get_predictions(Z):
    return np.round(Z)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size

def gradient_descent(X, Y, layers, iterations, alpha):
    W1, b1, W2, b2 = init_params(layers)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, layers)
        if i%50 == 0:
            print("Iteration: ",i)
            predictions=get_predictions(A2)
            print("Accuracy: ", get_accuracy(predictions,Y))
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(W1, b1, W2, b2, index):
    prediction = make_predictions(np.array(X_train).T, W1, b1, W2, b2)[0][index]
    label = Y_train[index]
    print("Data: ",X_train[index])
    print("Prediction: ", prediction)
    print("Label: ", label)
  

ln=4
# X_train=[]
# Y_train=[]
# for i in range(ln):
#     a=np.random.choice([0,1])
#     b=np.random.choice([0,1])
#     X_train.append([a,b])
#     if a!=b:
#         Y_train.append(1)
#     else:
#         Y_train.append(0)

X_train=[[0,0],[1,0],[0,1],[1,1]]
Y_train=[0,1,1,0]


W1, b1, W2, b2=gradient_descent(np.array(X_train).T, np.array(Y_train), [2,3,1], 5000, 0.1)

test_prediction(W1, b1, W2, b2, 0)
test_prediction(W1, b1, W2, b2, 1)
test_prediction(W1, b1, W2, b2, 2)
test_prediction(W1, b1, W2, b2, 3)

print("W1: ", W1)
print("b1: ", b1)
print("W2: ", W2)
print("b2: ", b2)


# Ideal for or:
# W1:  [[1.03356688 1.08055339]]
# b1:  [[-0.24254896]]
# W2:  [[1.60281478]]
# b2:  [[-0.31305127]]

# Ideal for and:
# W1=np.array([[1.17066244, 1.03698477]])
# b1=np.array([[-0.99155781]])
# W2=np.array([[2.04933135]])
# b2=np.array([[-0.67559342]])

# ot Ideal for xor:
# W1=np.array([[ 2.20140428, -2.57726568]])
# b1=np.array([[-2.37503557]])
# W2=np.array([[1.95311354]])
# b2=np.array([[0.04560792]])


# _, _, _, A2=forward_prop(W1, b1, W2, b2, [0,1])
# print(get_predictions(A2)[0][0])