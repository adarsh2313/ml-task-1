import numpy as np
import activation

layer = [96,48,24,6,3]
L = len(layer)

def init_weights():
    weights = {}
    np.random.seed(0)
    for l in range(1,L):
        weights['W'+str(l)] = np.random.rand(layer[l],layer[l-1])*0.01
        weights['b'+str(l)] = np.zeros((layer[l],1),dtype=float)
    return weights

def forward(X,weights):
    cache = {}
    cache['A0'] = X
    for l in range(1,L-1):
        cache['Z'+str(l)] = np.dot(weights['W'+str(l)],cache['A'+str(l-1)]) + weights['b'+str(l)]
        cache['A'+str(l)] = activation.relu_frwrd(cache['Z'+str(l)])
    
    cache['Z'+str(L-1)] = np.dot(weights['W'+str(L-1)],cache['A'+str(L-2)]) + weights['b'+str(L-1)]
    cache['A'+str(L-1)] = activation.sigmoid_frwrd(cache['Z'+str(L-1)])
    return cache

def compute_cost(A,Y):
    m = Y.shape[1]
    loss = np.square(A-Y)/2
    cost = (1/m) * np.sum(loss)
    return cost

def backward(cache,Y,weights,alpha):
    
    dA = np.subtract(cache['A4'],Y)

    dZ = activation.sigmoid_bkwrd(dA)
    m = cache['A'+str(L-2)].shape[1]
    dW = np.dot(dZ,cache['A'+str(L-2)].T)/m
    db = np.sum(dZ,axis = 1, keepdims=True)/m
    dA = np.dot(weights['W'+str(L-1)].T,dZ)
    weights['W'+str(L-1)] = np.subtract(weights['W'+str(L-1)], alpha*dW)
    weights['b'+str(L-1)] = np.subtract(weights['b'+str(L-1)], alpha*db)

    for l in reversed(range(1,L-1)):
        dZ = activation.relu_bkwrd_fc(dA, cache['Z'+str(l)])
        m = cache['A'+str(l-1)].shape[1]
        dW = np.dot(dZ,cache['A'+str(l-1)].T)/m
        db = np.sum(dZ,axis = 1, keepdims=True)/m
        dA = np.dot(weights['W'+str(l)].T,dZ)
        weights['W'+str(l)] = np.subtract(weights['W'+str(l)], alpha*dW)
        weights['b'+str(l)] = np.subtract(weights['b'+str(l)], alpha*db)
    
    return weights,dA