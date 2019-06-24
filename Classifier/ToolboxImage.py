"""
    Classifier.py 

    Helper imaging functions
    TODO: add load_data which should take care of all preprocessing operations
"""
__author__= "Dorian Stoica"

from os import listdir
import numpy as np
from PIL import Image as Image

def load_images(path, sizeX, sizeY, flip=True):
    """
    Helper function used to load images
    Arguments:
        path - path to the image folder
        sizeX, sizeY - size of the images
        flip - indicates whether the flipped images should be loaded as well (defaults to True)

    Returns:
        loaded_images - np.array of shape(numOfLoadedImages, sizeX, sizeY, 3)
    """
    loaded_images=list()
    for filename in listdir(path):
        img = Image.open(path+'/'+filename).resize((sizeX, sizeY))
        img.convert(mode='L')
        img_data = np.asarray(img)
        loaded_images.append(img_data)
        
        if (flip):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.convert(mode='L')
            img_data = np.asarray(img)
            loaded_images.append(img_data)
    #print('Loaded '+str(len(loaded_images))+ ' pictures')
    return np.array(loaded_images)

def loadTrainingData(path,sizeX, sizeY):

    dutchPath=path+"\\nederlands"
    dutchPics = load_images(dutchPath,sizeX,sizeY,True)
    roPath = path+"\\romania"
    romanianPics = load_images(roPath,sizeX,sizeY,True)

    trainPics = np.concatenate((dutchPics,romanianPics),axis=0)
    #Flatten and normalize
    trainData = trainPics.reshape(trainPics.shape[0],-1).T
    trainData = trainData/255.

    #Construct the labels 0 - Nederlands, 1-Romania
    numOfDutchImages = dutchPics.shape[0]
    numOfRomanianImages = romanianPics.shape[0]
    numImages = trainData.shape[1]
    assert(numOfDutchImages+numOfRomanianImages == numImages)
    trainLabels =np.zeros(numImages, dtype=int).reshape(1,numImages)    
    trainLabels[0][numOfDutchImages:]=1
    assert(trainData.shape[1]==trainLabels.shape[1])

    return trainData, trainLabels
    
def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(256,256,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])] + " \n Class: " + classes[y[0,index]])