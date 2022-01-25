import numpy as np

def relu_frwrd(Z):
    return np.maximum(0,Z)

def sigmoid_frwrd(Z):
    return 1/(1 + np.exp(-Z))

def relu_bkwrd_fc(dA,Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <=0] = 0
    return dZ

def sigmoid_bkwrd(dA):
    s = sigmoid_frwrd(dA)
    dZ = s*(1-s)
    return dZ