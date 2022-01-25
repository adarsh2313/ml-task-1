import numpy as np

def flatten_forward(A_p):
    X =  np.ravel(A_p).reshape(A_p.shape[0], -1)
    return X.T

def flatten_backward(dA,x):
    return dA.reshape(x)