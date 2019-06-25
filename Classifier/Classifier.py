"""Classifier.py 

    Deep Neural Network Binary Classifier 
    
    Usage: 
        - instantiate the model with the desired architecture(simply pass an array with the layers dimensions)
        - fit the training data by calling "train"
        - evaluate the model via "predict"
    """
__author__= "Dorian Stoica"


import numpy as np
import NeuralNet
import matplotlib.pyplot as plt


class Model:

    def __init__(self,layer_dims,initMode):
        self.params = NeuralNet.initializeParameters(layer_dims,initMode)
        self.velocities = NeuralNet.initializeVelocities(layer_dims)
        self.numberOfLayers = len(layer_dims)-1
    

    def train2(self,X,Y,num_iterations,num_batches=1,learning_rate = 0.0075, regularization_factor=0.0,print_cost=False):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
    
        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector of shape (1, number of examples)        
        num_iterations -- number of iterations of the optimization loop
        num_batches -- number of batches to be used by gradient descent
        learning_rate -- learning rate of the gradient descent update rule        
        regularization_factor -- regularization factor used in L2 regularization
        print_cost -- if True, it prints the cost every 100 steps        
        """
        np.random.seed(1)
        costs=[]
        m = X.shape[1]
        parameters = self.params

        batched_data = np.array_split(X,num_batches,axis=1)
        batched_labels = np.array_split(Y,num_batches,axis=1)
        assert (len(batched_data) == len(batched_labels))
        assert (len(batched_data)==num_batches)
      
        # Gradient descent main loop
        for i in range(0,num_iterations):
                            
            #Loop batches
            for batchIdx in range(0,num_batches):              
                currBatch = batched_data[batchIdx]
                currLabels = batched_labels[batchIdx]
                assert(currBatch.shape[1] == currLabels.shape[1])

                # Forward propagation: (L-1) ReLU units + 1 Sigmoid unit
                AL, caches = NeuralNet.forwardPropagation(currBatch,parameters)

                # Compute cost
                cost = NeuralNet.computeCost(AL,currLabels,parameters,self.numberOfLayers, regularization_factor)

                # Backward propagation
                grads = NeuralNet.backwardPropagation(AL,currLabels,caches,regularization_factor)

                # Update parameters                
                self.velocities, parameters = NeuralNet.updateParametersWithMomentum(parameters, grads, learning_rate, self.velocities,0)               

            if (print_cost and i%100==0):
                print("Cost after iteration %i: %f" %(i,cost))
                costs.append(cost)
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate) + " Weight decay factor= "+str(regularization_factor))
        plt.show()
        self.params = parameters            


    def train(self,X, Y, num_iterations = 3000,  learning_rate = 0.0075, regularization_factor = 0.0, print_cost=False):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector of shape (1, number of examples)        
        num_iterations -- number of iterations of the optimization loop
        num_batches -- number of batches to be used by gradient descent
        learning_rate -- learning rate of the gradient descent update rule        
        print_cost -- if True, it prints the cost every 100 steps        
        """

        np.random.seed(1)
        costs = []                         # keep track of cost
        m = X.shape[1]                     # number of examples

        parameters = self.params

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = NeuralNet.forwardPropagation(X, parameters)

            # Compute cost.
            cost = NeuralNet.computeCost(AL,Y, parameters, self.numberOfLayers,regularization_factor)
                                      
            # Backward propagation.
            grads = NeuralNet.backwardPropagation(AL, Y, caches,regularization_factor)
 
            # Update parameters.
            parameters = NeuralNet.updateParameters(parameters, grads, learning_rate)
            self.params = parameters
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
            
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate) + " Weight decay factor= "+str(regularization_factor))
        plt.show()
  
    def predict(self,X, y):
        """
        This function is used to predict the results of a  L-layer neural network.
    
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
    
        Returns:
        p -- predictions for the given dataset X
        """
        parameters = self.params
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
    
        # Forward propagation
        probas, caches = NeuralNet.forwardPropagation(X, parameters)
    
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
    
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Accuracy: "  + str(np.sum((p == y)/m)))
        
        return p

