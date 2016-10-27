#!/usr/bin/env python
'''
test.py
--------------------------------------------------------------------------------------------------------------------------
author: Brandon Radosevich
date: October 16, 2016
class: New Mexico State University EE565/490
project: Project 5
description:
    A Module to test accuracy of a given data model for MNIST Data set.
--------------------------------------------------------------------------------------------------------------------------
'''
#Libraries
#Standard Libraries
import argparse
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
#Math & Scientific Libraries
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier #Neural Network
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mnist import MNIST
import theano
import theano.tensor as T
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
class Classify(object):

    def __init__(self,filename, folder, nType):
        model = self.loadDataModel(filename)
        test_images, test_labels = self.loadMNISTDataSet(folder)
        if nType == "Convo":
            preds = self.classify(test_images, model)
            self.confusion_matrix(preds,test_labels)
        if nType == "Simple":
            preds = self.classifySimple(model,test_images)
            self.confusion_matrix(preds,test_labels)

    '''
    function: loadMNISTDataSet
    date: October 22, 2016
    description: This function loads the MNIST Data Set into the program
    '''
    def loadMNISTDataSet(self,folder):
        mndata = MNIST(folder)
        mndata.load_testing()
        test_images = np.array(mndata.test_images, dtype=float)
        test_images = test_images.reshape((-1,1,28,28)) # reshape to 28x28 pixel
        test_labels = np.array(mndata.test_labels, dtype=np.uint8)
        return test_images, test_labels

    '''
    function: loadDataModel
    date: October 22, 2016
    description: This function loads the Neural Network model into the program
    '''
    def loadDataModel(self, filename):
        with open(filename, 'rb') as f:
            mnist_model = pickle.load(f)
        f.close()
        return mnist_model

    '''
    function: classify
    date: October 24, 2016
    description: Finds predictions for a convolutional neural network given a data model
    '''
    def classify(self,test_images, model):
        preds = model.predict(test_images)
        return preds

    '''
    function: classifySimply
    date: October 24, 2016
    description: Finds predictions for a simple neural network given a data model
    '''
    def classifySimple(self,model,test_images):
        test = test_images.reshape((-1,784))
        #test = test_images / 255.0
        print test.shape
        preds = model.predict(test)
        return preds

    '''
    function: confusion_matrix
    date: October 12, 2016
    description: Prints a confusion matrix with the predictions and actual values for a given model
    '''
    def confusion_matrix(self,preds,test_labels):
        cm = confusion_matrix(test_labels, preds)
        plt.matshow(cm)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MNIST Training")
    parser.add_argument("-f","--file", dest="folder", help="The file to read data from", metavar="FOLDER",required=True)
    parser.add_argument("-t","--trainingFile", dest="trainingFile", help="The file to read data from", metavar="FILE",required=True)
    parser.add_argument("-n","--nType", dest="nType", help="The file to read data from", metavar="FILE",required=True)
    args = parser.parse_args()
    #lm = mnistTraining(args.trainingFile)
    Classify(args.trainingFile,args.folder,args.nType)
