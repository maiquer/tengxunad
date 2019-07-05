# -*- coding: utf-8 -*-
import numpy as np

y_true = np.array([1,2,3,4,5,6,7,8,9])
y_pre = np.array([1.2,2.3,3.5,4.3,4.6,5.6,6.1,7.1,8.8])

def L1(y_true,y_pre):
    return np.sum(np.abs(y_true-y_pre))
result = L1(y_true,y_pre)
print('L1 loss is {}'.format(result))

def L2(y_true,y_pre):
    return np.sum(np.square(y_true-y_pre))
result = L2(y_true,y_pre)
print('L2 loss is {}'.format(result))

def softmax(y_pre):
    return np.exp(y_pre)/np.sum(np.exp(y_pre))
z = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
y_pre = softmax(z)
y_true = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

def cross_entropy(y_true,y_pre):
    return -np.sum(y_true*np.log(y_pre))
print('softmax loss is {}'.format(cross_entropy(y_true,y_pre)))