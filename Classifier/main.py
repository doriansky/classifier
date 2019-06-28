"""
TODO list:

0.  Gather more data
    0a. Having more data -> split it into train and cross-validation 
    0b. Implement "early stopping"

1. Evaluate the model during training on a cross validation data set

2. Extend the regularization options : allow L1,L2 and dropout regularization

3. Properly document "train" : use mini-batch gradient descent with default num-batches=1 (aka batch gradient descent)
    3a. decay the learning rate during mini-BGD : see Goodfellow page 295 eq. 8.14

4. Extend the optimization options: Momentum, NesterovMomentum, AdaGrad,Adadelta, RMSprop, Adam, Adamax, Nadam, AMSGrad
For momentum, see Goodfellow pg 296

5. Unit tests 

DONE
0. Figure out if AdaGrad and RMSProp are correctly implemented : in the update rule , np.sum of the current squared gradient is needed,
otherwise parameters will explode (due to the division of grads (~e-3) with the sqrt of accum. squared gradients (~e-8) ....resulting in very big numbers)

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


#No momentum :(
"""
model = Classifier.Model(layer_dims,initMode="xavier")
model.train(train_data, train_labels,num_iterations=1000,learning_rate=0.0075,regularization_factor=0.2,print_cost=True)
model.predict(test_data,test_labels)

#Classical momentum
model2 = Classifier.Model(layer_dims,initMode="xavier")
model2.train2(train_data, train_labels,num_iterations=1000,num_batches=1,learning_rate=0.0075,regularization_factor=0.2,momentum=0.8,print_cost=True)
model2.predict(test_data,test_labels)

#Nesterov momentum
model3 = Classifier.Model(layer_dims,initMode="xavier")
model3.train3(train_data, train_labels,num_iterations=1000,num_batches=1,learning_rate=0.0075,regularization_factor=0.2,momentum=0.8,print_cost=True)
model3.predict(test_data,test_labels)

#Adagrad
model4 = Classifier.Model(layer_dims,initMode="xavier")
model4.train4(train_data, train_labels,num_iterations=1000,num_batches=1,learning_rate=0.75,regularization_factor=0.2,print_cost=True)
model4.predict(test_data,test_labels)

#RMSProp
model5 = Classifier.Model(layer_dims,initMode="xavier")
model5.train5(train_data, train_labels,num_iterations=1000,num_batches=1,learning_rate=0.0075,regularization_factor=0.2,decayRate=0.9,print_cost=True)
model5.predict(test_data,test_labels)

#RMSProp with Nesterov
model6 = Classifier.Model(layer_dims,initMode="xavier")
model6.train6(train_data, train_labels,num_iterations=1000,num_batches=1,learning_rate=0.0075,regularization_factor=0.2,momentum=0.8,decayRate=0.9,print_cost=True)
model6.predict(test_data,test_labels)
"""


#Adam
model7 = Classifier.Model(layer_dims,initMode="xavier")
model7.train7(train_data, train_labels,num_iterations=1000,num_batches=1,learning_rate=0.0075,regularization_factor=0.2,firstDecayRate=0.1,secondDecayRate=0.001,print_cost=True)
model7.predict(test_data,test_labels)
