import numpy as np

def forward(A_p,f,stride):

    # Retrieving the shape parameters and calculating n_H and n_W
    (m, n_H_p, n_W_p, n_C_p) = A_p.shape
    n_H = (n_H_p  -f)//stride + 1
    n_W = (n_W_p  -f)//stride + 1
    n_C = n_C_p
    
    A = np.zeros((m, n_H, n_W, n_C))

    # Applying maxpool
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Calculating the parameters for slicing the input 
                    h_f = h*stride
                    h_t = h_f + f
                    w_f = w*stride
                    w_t = w_f + f

                    A[i, h, w, c] = np.max(A_p[i, h_f:h_t, w_f:w_t, c])

    cache_maxpool = (A_p,f,stride)

    return A,cache_maxpool

def backward(dA, cache_maxpool):     
    # Getting the required variables and initialising with zero matrix for the output
    (A_prev,stride,f) = cache_maxpool
    m, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Getting parameters for slicing the input layer for convolving
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Computing backpropagation
                    a_prev_slice = A_prev[i,vert_start:vert_end, horiz_start:horiz_end, c]
                    mask = a_prev_slice == np.max(a_prev_slice)
                    dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])

    assert(dA_prev.shape == A_prev.shape)
    return dA_prev