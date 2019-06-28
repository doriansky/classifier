"""
    NeuralNet.py 

    Implementation of the neural network building blocks: initialize parameters(weights and biases), 
                                                          forward propagate the activations,
                                                          compute cross-entropy cost,
                                                          backward propagate the gradients,
                                                          update the params via gradient descent                                                        
    """
__author__= "Dorian Stoica"

import numpy as np
import ToolboxMath

def initializeMomentEstimates(layer_dims):

    """
    Arguments:
    layer_dims -- python array(list) containing the dimensions of each layer
    init_mode -- string specifying the initialization mode : random or xavier
    Returns:
    parameters -- python dict containing the parameters "W1","b1",...."WL","bL":
        Wl -- weight matrix of shape (layer_dims[l],layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l],1)
    """

    firstMomentEstimates = {}
    secondMomentEstimates = {}
    L = len(layer_dims)    
    for l in range(1,L):
        firstMomentEstimates['W'+str(l)] = np.zeros(shape=(layer_dims[l],layer_dims[l-1]))
        firstMomentEstimates['b'+str(l)] = np.zeros(shape=(layer_dims[l],1))
        secondMomentEstimates['W'+str(l)] = np.zeros(shape=(layer_dims[l],layer_dims[l-1]))
        secondMomentEstimates['b'+str(l)] = np.zeros(shape=(layer_dims[l],1))

    return firstMomentEstimates, secondMomentEstimates


def initializeAccumulatedSquaredGradients(layer_dims):

    """
    Arguments:
    layer_dims -- python array(list) containing the dimensions of each layer
    init_mode -- string specifying the initialization mode : random or xavier
    Returns:
    parameters -- python dict containing the parameters "W1","b1",...."WL","bL":
        Wl -- weight matrix of shape (layer_dims[l],layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l],1)
    """

    acSqGradients = {}
    L = len(layer_dims)    
    for l in range(1,L):
        acSqGradients['W'+str(l)] = np.zeros(shape=(layer_dims[l],layer_dims[l-1]))
        acSqGradients['b'+str(l)] = np.zeros(shape=(layer_dims[l],1))

    return acSqGradients     

def initializeParameters(layer_dims,initMode):
    """
    Arguments:
    layer_dims -- python array(list) containing the dimensions of each layer
    init_mode -- string specifying the initialization mode : random or xavier
    Returns:
    parameters -- python dict containing the parameters "W1","b1",...."WL","bL":
        Wl -- weight matrix of shape (layer_dims[l],layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l],1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    if (initMode=="random"):
        for l in range(1,L):
            parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
            parameters['b'+str(l)] = np.zeros(shape=(layer_dims[l],1))
    elif (initMode=="xavier"):
        for l in range(1,L):
            parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(1/layer_dims[l-1])  # multiply by 2 ??
            parameters['b'+str(l)] = np.zeros(shape=(layer_dims[l],1))
    else:
        raise Exception("Invalid initialization mode !")

    return parameters        

def initializeVelocities(layer_dims):
    """
    Arguments:
    layer_dims -- python array(list) containing the dimensions of each layer    
    Returns:
    velocities -- python dict containing the zero-initialized velocities "v_W1","v_b1",...."v_WL","v_bL":
        v_Wl -- weight matrix of shape (layer_dims[l],layer_dims[l-1])
        v_bl -- bias vector of shape (layer_dims[l],1)
    """
    np.random.seed(1)
    velocities = {}
    L = len(layer_dims)
    for l in range(1,L):
        velocities['W'+str(l)] = np.zeros(shape=(layer_dims[l], layer_dims[l-1]))
        velocities['b'+str(l)] = np.zeros(shape=(layer_dims[l], 1))
    return velocities

def computeLinearActivations(A,W,b):
    """
    Implementation of the linear part of forward propagation
    
    Arguments:
    A -- activations from previous layer (or input data) : (size of previous layer, number of examples)
    W -- weights matrix : np array of shape (Size of current layer, size of previous layer)
    b -- bias vector, np array of shape (size of current layer, 1)
    
    Returns:
    Z -- input of the activation function (aka pre-activation)
    cache -- python dict containing A,W and b; useful during backprop
    """
    Z = np.dot(W,A)+b
    cache = (A,W,b)
    
    return Z,cache

def computeNonLinearActivations(A_prev, W, b, activation):
    """
    Implementation of the forward propagation for one layer
    Arguments:
    A_prev -- activations from previous layer (size of previous layer, num of examples)
    W -- weights matrix : np array of shape (Size of current layer, size of previous layer)
    b -- bias vector :  np array of shape (size of current layer,1)
    activation -- the activation to be used in this layer. String that can be "sigmoid" or "relu"
    
    Returns:
    A -- the output of the activation function, aka post-activation value
    cache -- python dict containing "linear_cache" and "activation_cache"
    """
    
    if activation == "sigmoid":
        Z,linear_cache = computeLinearActivations(A_prev,W,b)
        A, activation_cache = ToolboxMath.sigmoid(Z)
        
    elif activation == "relu":
        Z,linear_cache = computeLinearActivations(A_prev,W,b)
        A,activation_cache = ToolboxMath.relu(Z)
        
    cache = (linear_cache, activation_cache)
    return A,cache

def forwardPropagation(X,parameters):
    """
    Implementation of the forward propagation for L layers ((L-1) relu and one sigmoid)
    
    Arguments:
    X -- data , np array of shape (input size, number of examples)
    parameters -- output of initalize_paramters_deep
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing every cache of linear_activation_forward()
    """
    
    caches = []
    A = X
    L = len(parameters) // 2 # divide by 2 since for each layer we have 2 params (W and b)
    
    for l in range(1,L):
        A_prev = A
        A, cache = computeNonLinearActivations(A_prev,parameters['W'+str(l)], parameters['b'+str(l)],"relu")
        caches.append(cache)
    
    AL,cache = computeNonLinearActivations(A, parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
    caches.append(cache)
    
    return AL, caches

def computeCost(AL,Y, parameters, numLayers, regularizationFactor=0.0):
    """
    Implement the cost function
    
    Arguments:
    AL -- Probability vector corresponding to the label predictions , shape(1,number of examples)
    Y -- true label vector, shape(1,number of examples)
    parameters -- dictionary containing the model weights and biases. Useful for L1 or L2 regularization
    numLayers -- number of layers.     
    regularizationFactor -- regularization factor 
    
    Returns:
    cost -- regularized cross-entropy cost
    """

    m = Y.shape[1]
    cost = -np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL)))/m    
    
    L = len(parameters)//2
    regularizationTerm = 0.0
    for l in range(1,numLayers+1):
        W = parameters['W'+str(l)]
        regularizationTerm = regularizationTerm+np.sum(np.square(W));
    
    regularizationTerm = (regularizationFactor/(2*m))*regularizationTerm
    cost = cost+regularizationTerm
    cost = np.squeeze(cost)  

    return cost

def computeGradients(dA, cache, activation, regularization_factor=0.0):
    """
    Implementation of the backward propagation for the linear->activation layer
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache)
    activation -- the activation to be used in this layer
    
    Returns:
    dA_prev -- Gradient of the cost w.r.t the activation of the previous layer l-1 (same shape as A_prev)
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
  
    linear_cache,activation_cache = cache
    if activation == "relu":
        dZ = ToolboxMath.relu_backward(dA, activation_cache)              
    elif activation == "sigmoid":
        dZ = ToolboxMath.sigmoid_backward(dA, activation_cache)
         
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = (1/m)*np.dot(dZ, A_prev.T)+(regularization_factor/m)*W #back-prop with regularization
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    return dA_prev,dW,db

def backwardPropagation(AL,Y,caches, regularization_factor=0.0):
    """
    Implementation of the backward propagation for the (L-1) relu units and 1 sigmoid unit network
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true label vector
    caches -- list of caches containing every cache of linear_activation_forward with relu (l in range(L-1))
                and the cache of linear_activation_forward with sigmoid  (caches[L-1])
                
    Returns:
    grads -- A dictionary with the gradients
        grads["dA"+str(l)] = ...
        grads["dW"+str(l)] = ...
        grads["db"+str(l)] = ...                
    """
    grads={}
    L = len(caches) #number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))   
    current_cache = caches[L-1]
    grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)] = computeGradients(dAL,current_cache,"sigmoid",regularization_factor)
    
    #Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache=caches[l]
        dA_prev_temp, dW_temp, db_temp = computeGradients(grads["dA"+str(l+1)],current_cache,"relu",regularization_factor)
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
    
    return grads

def updateParameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent    
    Arguments:
    parameters -- python dictionary containing the parameters 
    grads -- python dictionary containing the gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing the updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network
   
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db" + str(l+1)]

    return parameters

def updateParametersWithMomentum(parameters, grads, learningRate, velocities, momentum):
    """
    Update parameters using gradient descent with momentum. The update of params is smoothed out by considering the past gradients. 
    The momentum algorithm accumulates an exponentially decaying moving average of past gradients and continues to move in their direction.

    Arguments:
    parameters -- python dictionary containing the parameters 
    grads -- python dictionary containing the gradients, output of L_model_backward
    learningRate -- the learning rate 
    velocities -- current velocities
    moementum -- momentum term. Setting this to zero will result in applying the standard update rule

    Returns:
    parameters -- python dictionary containing the updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...

    velocities -- python dictionary containing the updated velocities (to be used at the next iteration)
                    velocities["v_W"+str(l)] = ....
                    velocities["v_b"+str(l)] = ....
    """

    L = len(parameters)//2

    # Below update rules are compatible with both classical momentum as well as with Nesterov momentum
    # For Nesterov momentum the "grads" are already computed at this stage using the so-called "interimParams". 
    # The interimParams are computed in the main gradient descent loop, before starting the forward-backward iteration
    
    # Update rules for epoch "t"
    # v(t) = m*v(t-1)-k*grads
    # params(t) = params(t-1)+v(t)

    # Goodfellow page 296, section 8.3.2
    # https://dominikschmidt.xyz/nesterov-momentum/
    # https://medium.com/konvergen/momentum-method-and-nesterov-accelerated-gradient-487ba776c987
    #First update the velocities    
    for l in range(L):
        # Goodfellow eq. 8.15, pg. 296
        velocities["W"+str(l+1)] = momentum*velocities["W"+str(l+1)]-learningRate*grads["dW"+str(l+1)]
        velocities["b"+str(l+1)] = momentum*velocities["b"+str(l+1)]-learningRate*grads["db"+str(l+1)]

    #Now apply the update rule for parameters
    for l in range(L):
        #Goodfellow eq. 8.16 pg. 296
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]+velocities["W"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]+velocities["b"+str(l+1)]

    return velocities,parameters

def updateParametersWithAdaptiveLearningRate(parameters, grads, learningRate,squaredGradientSum):
    """
    Update parameters with adaptive learning rate :this method is compatible with both Adagrad and RMSProp
    The only difference between Adagrad and RMS is the way in which the gradients are accumulated...but the actual
    update rule is identical

    Arguments:
    parameters -- python dictionary containing the parameters 
    grads -- python dictionary containing the gradients, output of L_model_backward
    accumulatedSquaredGradients -- python np.array containing the accumulated squared gradients for each layer
    Returns:
    parameters -- python dictionary containing the updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network
    eps = 1e-8
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learningRate * np.divide(grads["dW"+str(l+1)],eps+np.sqrt(np.sum(squaredGradientSum["W"+str(l+1)])))
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learningRate * np.divide(grads["db"+str(l+1)],eps+np.sqrt(np.sum(squaredGradientSum["b"+str(l+1)])))

    return parameters

def updateParametersWithAdaptiveLearningRateAndMomentum(parameters, grads, learningRate, velocities, momentum, squaredGradientSum):
    L = len(parameters) // 2 # number of layers in the neural network
    eps = 1e-8

    for l in range(L):
        # Goodfellow , pg. 310, RMSProp with momentum
        velocities["W"+str(l+1)] = momentum*velocities["W"+str(l+1)]-learningRate*np.divide(grads["dW"+str(l+1)],eps+np.sqrt(np.sum(squaredGradientSum["W"+str(l+1)])))
        velocities["b"+str(l+1)] = momentum*velocities["b"+str(l+1)]-learningRate*np.divide(grads["db"+str(l+1)],eps+np.sqrt(np.sum(squaredGradientSum["b"+str(l+1)])))

    #Now apply the update rule for parameters
    for l in range(L):
        #Goodfellow eq. 8.16 pg. 296
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]+velocities["W"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]+velocities["b"+str(l+1)]

    return velocities, parameters

