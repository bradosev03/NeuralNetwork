#!/usr/bin/env python
'''
train.py
--------------------------------------------------------------------------------------------------------------------------
author: Brandon Radosevich
date: October 16, 2016
class: New Mexico State University EE565/490
project: Project 5
description:
    A Module to implement a Convolutional Neural Network and a Simple Neural Network for use on the classic MNIST Dataset.
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

class mnistTraining(object):

    def __init__(self,folder,filename,type):
        self.filename = filename
        if (type == "Convo"):
            train_images, train_labels, test_images, test_labels = self.load_dataset(folder)
            self.cnnTraining(train_images, train_labels, test_images, test_labels)
        if type == "Simple":
            training, testing = self.loadData(folder)
            self.training(training,testing)
        if type == "Show":
            self.printDataSet()

    '''
    function: load_dataset
    date: October 12, 2016
    description: Loads dataset from MNIST Folder. For use in Simple Neural Network
    '''
    def loadData(self, folder):
        mndata = MNIST(folder)
        mndata.load_training()
        mndata.load_testing()
        train_images = np.array(mndata.train_images, dtype=float)
        train_labels = np.array(mndata.train_labels, dtype=int)
        test_images = np.array(mndata.test_images, dtype=float)
        test_labels = np.array(mndata.test_labels, dtype=int)
        training = [train_images,train_labels]
        testing = [test_images,test_labels]
        return (training,testing)

    '''
    function: load_dataset
    date: October 12, 2016
    description: Loads dataset from MNIST Folder. For use in Convolutional Neural Network
    '''
    def load_dataset(self,folder):
        mndata = MNIST(folder)
        mndata.load_training()
        mndata.load_testing()
        train_images = np.array(mndata.train_images, dtype=float)
        train_images = train_images.reshape((-1,1,28,28)) # reshape to 28x28 pixel
        train_labels = np.array(mndata.train_labels, dtype=np.uint8)
        test_images = np.array(mndata.test_images, dtype=float)
        test_images = test_images.reshape((-1,1,28,28)) # reshape to 28x28 pixel
        test_labels = np.array(mndata.test_labels, dtype=np.uint8)
        print 'Training Images Shape: ',train_images.shape
        print 'Training Labels Shape: ',train_labels.shape
        print 'Test Images Shape: ',test_images.shape
        print 'Test Labels Shape: ',test_labels.shape
        return train_images, train_labels, test_images, test_labels

    '''
    function: cnnTraining
    date: October 12, 2016
    description: Trains a Convolutional Neural Network on the MNIST Data Set
    '''
    def cnnTraining(self,train_images,train_labels, test_images, test_labels):
        net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 1, 28, 28),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # layer maxpool1
        maxpool1_pool_size=(2, 2),
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,
        # dense
        dense_num_units=256,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.04,
        max_epochs=50,
        verbose=1,)
        nn = net1.fit(train_images, train_labels)
        print 'Saving File: '
        with open(self.filename, 'wb') as output:
            pickle.dump(nn,output, pickle.HIGHEST_PROTOCOL)
        print "[+] Saving Completed"

    '''
    function: training
    date: October 12, 2016
    description: Trains a Simple Neural Network on the MNIST Data Set
    '''
    def training(self,training,testing):
        train_images = training[0]
        x_training = training[0] / 255.0
        y_training = training[1]
        x_test = testing[0] / 255.0
        y_test = testing[1]
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=1e-4,
            solver='sgd', verbose=10, tol=0.0001, random_state=1,
            learning_rate_init=.1,learning_rate="adaptive")
        mlp.fit(x_training,y_training)
        print mlp.get_params()
        with open(self.filename, 'wb') as output:
            pickle.dump(mlp,output, pickle.HIGHEST_PROTOCOL)
        print "[+] Saving Completed"

    '''
    function: printDataSet
    date: October 12, 2016
    description: Prints sample digits from the MNIST dataset.
    '''
    def printDataSet(self):
        digits = datasets.load_digits()
        images_and_labels = list(zip(digits.images, digits.target))
        for index, (image, label) in enumerate(images_and_labels[0:8]):
            plt.subplot(2, 4, index+1)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('%i' % label)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MNIST Training")
    parser.add_argument("-f","--file", dest="trainingFile", help="The file to read data from", metavar="FILE",required=True)
    parser.add_argument("-n","--name", dest="filename", help="The file to save data model to", metavar="FILE",required=True)
    parser.add_argument("-t","--type", dest="type", help="The Type of Neural Network to Use", metavar="FILE",required=True)
    args = parser.parse_args()
    mnistTraining(args.trainingFile,args.filename,args.type)
