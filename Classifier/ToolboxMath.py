"""
    ToolboxMath.py 

    Helper math functions    
"""
__author__="Dorian Stoica"


import numpy as np

def sigmoid(Z):
    """
    Implementation of the sigmoid function
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns z as well, useful during backprop
    """
    A=1/(1+np.exp(-Z))
    cache = Z
    return A,cache

def sigmoid_backward(dA, cache):
    """
    Implementation of the derivative of the sigmoid function
    
    Arguments:
    dA -- postactivation gradient, of any shape
    cache -- Z
    
    Returns:
    dZ -- gradient of the cost with respect to Z
    """
    Z = cache
    s=1/(1+np.exp(-Z))
    dZ = dA*s*(1-s) #chain-rule
    
    assert(dZ.shape == Z.shape)
    return dZ

def relu(Z):
    """
    Implementation of the RELU function
    
    Arguments:
    Z -- output of the linear layer, of any shape
    
    Returns:
    A -- post-activation parameter, same shape as Z
    cache -- python dictionary containing "Z". 
    """
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A,cache

def relu_backward(dA, cache):
    """
    Implementation of the derivative of RELU
    
    Arguments:
    dA -- postactivation gradient, of any shape
    cache -- Z
    
    Returns:
    dZ -- gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    
    assert(dZ.shape == Z.shape)
    return dZ