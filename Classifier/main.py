"""
TODO list:

0.  Gather more data
    0a. Having more data -> split it into train and cross-validation 
    0b. Implement "early stopping"

1. Evaluate the model during training on a cross validation data set
2. Extend the regularization options : allow L1,L2 and dropout regularization
3. Properly document "train" : use mini-batch gradient descent with default num-batches=1 (aka batch gradient descent)
    3a. decay the learning rate during mini-BGD : see Goodfellow page 295 eq. 8.14
4. Extend the optimization options: SGD, Momentum, NesterovMomentum, AdaGrad,Adadelta, RMSprop, Adam, Adamax, Nadam, AMSGrad
For momentum, see Goodfellow pg 296




DONE
1. extend the Neural network; allow Xavier initialization of the parameters
1a. during initialization , self.noLayers should be initialized as well

4. the learned parameters should be encapsulated in the Model 
       4a. "train" should not return the params, instead these should be stored in a member
       4b. "predict" will no longer require "params" as an argument since the NN Model is already trained at that time

"""

import ToolboxImage
import Classifier
import numpy as np

imgSize = 128
path = "C:\\Users\\dorian.stoica\\Desktop\\house classifier\\training_data\\"
train_data, train_labels = ToolboxImage.loadTrainingData(path,imgSize,imgSize)
path = "C:\\Users\\dorian.stoica\\Desktop\\house classifier\\test_data\\"
test_data, test_labels = ToolboxImage.loadTrainingData(path,imgSize,imgSize)

n_x = train_data.shape[0]
n_h = 20 #size of the hidden layer
n_y = 1
layer_dims=(n_x,50,n_h,n_y)


model = Classifier.Model(layer_dims,initMode="xavier")
model.train2(train_data, train_labels,num_iterations=1000,num_batches=1,learning_rate=0.0075,regularization_factor=0.2,print_cost=True)
model.predict(test_data,test_labels)

model2 = Classifier.Model(layer_dims,initMode="xavier")
model2.train(train_data, train_labels,num_iterations=1000,learning_rate=0.0075,regularization_factor=0.2,print_cost=True)
model2.predict(test_data,test_labels)

print("done")