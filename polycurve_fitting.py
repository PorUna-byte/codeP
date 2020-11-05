# -*- coding:utf-8 -*- #
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from random import *
def calX(data_amount,order):
    X=np.zeros((data_amount,order+1))
    for i in range(data_amount):
        x_i= ((float)(i * 6) / data_amount)+gauss(0,0.12)
        for j in range(0, order + 1, 1):
                X[i][j] = x_i ** j
    return X
def calY(data_amount):
    Y=np.zeros((data_amount,1))
    for i in range(data_amount):
            Y[i][0] = sin(float(pi * 2 * i) / data_amount)+gauss(0,0.12)
    return Y

def calloss_nopunish_gradient(X,Y,W,data_amount):
    A=np.dot(X,W)-Y
    return 1/(float)(data_amount)*np.dot(X.T,A)
def calloss_withpunish_gradient(X,Y,W,data_amount,lam):
    A=np.dot(X,W)-Y
    return 1/(float)(data_amount)*np.dot(X.T,A)+lam*W

def calloss_nopunish(X,Y,W,data_amount):
    A=Y-np.dot(X,W)
    return 1/(float)(2*data_amount)*np.dot(A.T,A)
def calloss_withpunish(X,Y,W,data_amount,lam):
    A=Y-np.dot(X,W)
    return 1/(float)(2*data_amount)*np.dot(A.T,A)+lam/2*np.dot(W.T,W)

def calw_nopunish_accurate(X, Y):
    A = np.dot(X.T, X)
    B = np.dot(X.T, Y)
    inv_A = np.linalg.inv(A)
    return np.dot(inv_A, B)


def calw_withpunish_accurate(X, Y, data_amount, lam, order):
    A = np.dot(X.T, X)
    B = np.identity(order + 1) * lam * (data_amount)
    C = np.dot(X.T, Y)
    return np.dot(np.linalg.inv(A + B), C)

def gradient_descent_nopunish(X,Y,data_amount,order,alpha):
   # W = np.ones((order+1,1))   #初始W,可任选一个
    W = np.array([-0.12,1.91,-0.94,0.105]).reshape(4,1)
    gradient = calloss_nopunish_gradient(X, Y, W,data_amount)
    while not np.dot(gradient.T,gradient) <= 1e-9:
        W = W - alpha * gradient
        gradient = calloss_nopunish_gradient(X,Y,W,data_amount)
        print(W)
    return W

def gradient_descent_withpunish(X,Y,data_amount,order,alpha,lam):
    #W = np.ones((order+1,1))
    W = np.array([-0.12, 1.91, -0.94, 0.105]).reshape(4, 1)
    gradient = calloss_withpunish_gradient(X,Y,W,data_amount,lam)
    while not np.dot(gradient.T,gradient) <= 1e-9:
        W = W - alpha * gradient
        gradient = calloss_nopunish_gradient(X,Y,W,data_amount)
        print(W)
    return W

def conjugate_gradient_nopunish(X,Y,order):
    A=np.dot(X.T,X)
    b=np.dot(X.T,Y)
    W=np.ones((order+1,1))
    r=b-np.dot(A,W)
    p=r
    while True:
        alpha = np.dot(r.T, r) / np.dot(np.dot(p.T, A), p)
        W = W +alpha*p
        r_ = r - np.dot(alpha*A, p)
        if np.dot(r_.T,r_) <= 1e-8:
            break
        beta=np.dot(r_.T,r_)/np.dot(r.T,r)
        p=r_+beta*p
        r = r_
        print(W)
    return W

def conjugate_gradient_withpunish(X,Y,order,data_amount,lam):
    A=np.dot(X.T,X)+(data_amount)*lam*np.identity(order + 1)
    b=np.dot(X.T,Y)
    W=np.ones((order+1,1))
    r=b-np.dot(A,W)
    p=r
    while True:
        alpha = np.dot(r.T, r) / np.dot(np.dot(p.T, A), p)
        W = W +alpha*p
        r_ = r - np.dot(alpha*A, p)
        if np.dot(r_.T,r_) <= 1e-8:
            break
        beta=np.dot(r_.T,r_)/np.dot(r.T,r)
        p=r_+beta*p
        r = r_
        print(W)
    return W
def plotcurve():
    plt.figure(figsize=(48, 48))
    for k in range(27):
        X = calX(20 * (k + 1), k / 3 + 1)
        Y = calY(20 * (k + 1))
        x_ = np.arange(0, 6, 0.01)
        W = calw_nopunish_accurate(X, Y)
        y_ = []
        for i in x_:
            array = []
            for j in range(k / 3 + 2):
                array.append(i ** j)
            y_.append(np.dot(W.T, np.array(array)))
        plt.subplot(9, 3, 0 + k + 1)
        plt.plot(x_, y_)
        plt.scatter(X[0:, 1:2], Y, color='m', marker='.')  # 画散点图
        plt.legend()
    plt.show()
    
