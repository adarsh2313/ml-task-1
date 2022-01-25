import numpy as np

def init_filters():
    
    F1 = np.random.randn(3,3,1,3)*0.1            # Filter1 of size 3x3x1x4
    b1 = np.zeros((1,1,1,3),dtype=float)         # bias for Filter1
    F2 = np.random.randn(3,3,3,6)*0.1            # Filter2 of size 3x3x3x6
    b2 = np.zeros((1,1,1,6),dtype=float)         # bias for Filter2
    
    return F1,b1,F2,b2

def padwithzeros(X,pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values = 0)
    return X_pad

def forward(A_p,F,b,stride,pad):

    # Retrieving the dimensions of A_prev and filter
    (m, n_H_p, n_W_p, _) = A_p.shape
    (f, _, _, n_C) = F.shape

    # Calculating n_H and n_W and initialising the output matrix
    n_H = (n_H_p + 2*pad -f)//stride + 1
    n_W = (n_W_p + 2*pad -f)//stride + 1
    Z = np.zeros((m,n_H,n_W,n_C))

    # Padding the input layer with zeros
    A_p_pad = padwithzeros(A_p,pad)

    # Convolving the input with given filter
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    # Calculating the parameters for slicing the input 
                    h_f = h*stride
                    h_t = h_f + f
                    w_f = w*stride
                    w_t = w_f + f
                    
                    # Slicing and convolving
                    A_slice = A_p_pad[i, h_f:h_t ,w_f:w_t,:]
                    Z[i,h,w,c] = np.sum(np.multiply(A_slice,F[...,c])+b[...,c])
    
    cache_conv = (A_p,F,b,stride,pad)
    return Z, cache_conv

def backward(dA, cache_conv):        

    # Getting the required variables for backpropagation
    (A_prev, W, b, stride, pad) = cache_conv
    (_, n_H_prev, n_W_prev, _) = A_prev.shape
    (f, f, n_C_prev, _) = W.shape
    (m, n_H, n_W, n_C) = dA.shape

    # Initialise 0 for the outputs with correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    dF = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = padwithzeros(A_prev, pad)
    dA_prev_pad = padwithzeros(dA_prev, pad)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    # Getting parameters for slicing the input layer for convolving
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Slicing and then calculating the gradients
                    a_slice = A_prev_pad[i,vert_start:vert_end, horiz_start:horiz_end, :]
                    dA_prev_pad[i,vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dA[i, h, w, c]
                    dF[:,:,:,c] += a_slice * dA[i, h, w, c]
                    db[:,:,:,c] += dA[i, h, w, c]

        dA_prev[i, :, :, :] = dA_prev_pad[i,pad:-pad, pad:-pad, :]    # Unpadding dA_prev
    
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
   
    return dA_prev, dF, db

def updatefilters(F,b,dF,db,alpha):
    F -= alpha*dF
    b -= alpha*db
    return F,b                           