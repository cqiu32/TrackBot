# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:13:47 2016

@author: badarim
"""

# This is just a sandbox file I'm creating to start of experimenting with different algorithms

# I'm using Knn for starters; The plan is to try EKF or others using http://filterpy.readthedocs.org/en/latest/kalman/ExtendedKalmanFilter.html

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation

import random as ran

def error(l1, l2):
    return sum((c - a)**2 + (d - b)**2 for ((a, b), (c, d)) in zip(l1, l2))**0.5

f = open('training_data.txt');
data =np.loadtxt(f, delimiter = ',');

n_neighbors = 7

labelXArray = []
labelYArray = []
labels = []
trainingData = []
noOfFramesToPredict = 60
for i in range(len(data) - noOfFramesToPredict-1):
    val = data[i]
    
    xvel = 0.
    yvel = 0.
    if i != 0:
        prev = data[i-1]
        xvel = 30 * (data[i][0] - prev[0])
        yvel = 30 * (data[i][1] - prev[1])
    trainingData.append([val[0], val[1], xvel, yvel])
    labelsX = []
    labelsY = []
    for j in range(1, noOfFramesToPredict+1):
        # get next 60 frames of x, y positions as labels for regression.
        labelsX.append(data[j][0])
        labelsY.append(data[j][1])        
    labelXArray.append(labelsX)
    labelYArray.append(labelsY)

print len(trainingData), len(labelXArray), len(labelYArray)
#for i in range(len(data)):
#    if (i + n_neighbors) < len(data):
#        val = data[i+n_neighbors];
#        labelXArray.append(val[0]);
#        labelYArray.append(val[1]);        
#        labels.append(val);
#        trainingData.append(data[i]);

#print len(labels), len(data), len(trainingData)

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
#                                                                     labelXArray,
#                                                                     test_size = 0.2, 
#                                                                     random_state = 0)
#print len(X_train), len(y_train), len(X_test), len(y_test)
# create an array for every frame
modelsX = []
for i in range(noOfFramesToPredict):
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
    temp = [yt[i] for yt in labelXArray]
    neigh.fit(trainingData, temp)
    modelsX.append(neigh)
   
    

# knn model for x
#neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
#neigh.fit(X_train, y_train)

#score = neigh.score(X_test,y_test)
#print "CV score(x) is ", score

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
#                                                                     labelYArray,
#                                                                     test_size = 0.2, 
#                                                                     random_state = 0)


modelsY = []
for i in range(noOfFramesToPredict):
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
    
    temp = [yt[i] for yt in labelYArray]
    neigh.fit(trainingData, temp)
    modelsY.append(neigh)
                                                                     
# knn model for y
#neighY = KNeighborsRegressor(n_neighbors=n_neighbors)
#neighY.fit(X_train, y_train)

#score = neighY.score(X_test,y_test)
#print "CV score(y) is ", score       
        
test1 = open('inputs/test01.txt');
test1Data =  np.loadtxt(test1, delimiter = ',');       
errors = []
NUM_INPUTS = 1
expectedPredictions = []
for i in range(NUM_INPUTS):
    index = ran.randint(0, len(test1Data) -1)
    features = []
    if index != 0:
        for k in range(1, noOfFramesToPredict+1):
            expectedPredictions.append([test1Data[index+k][0], test1Data[index+k][1]])
        velX = 30 * (test1Data[i][0] - test1Data[i-1][0])
        velY = 30 * (test1Data[i][1] - test1Data[i-1][1])
        features = [test1Data[i][0], test1Data[i][1], xvel, yvel]
        
        predictions = []
        predictions = []
        for m in range(len(modelsX)):
            modX = modelsX[m]
            modY = modelsY[m]
            predictions.append([modX.predict(features), modY.predict(features)])
        print "For row: ", test1Data[index]

        np.savetxt('predictions-knn-60.txt', predictions, delimiter=',', fmt='%10.3f');
        
import matplotlib.pyplot as plt

def plotGraph(arr1, arr2, label1, label2):
   plt.plot(np.array(arr1)[:,0], np.array(arr1)[:,1], 'ro')
   plt.axis([0,600, 0,600])
   
   plt.plot(np.array(arr2)[:,0], np.array(arr2)[:,1], 'bo')
   plt.legend(labels = [label1, label2])
   plt.show()  

plotGraph(predictions, expectedPredictions, 'predictions', 'expectedPredictions')

def plotLine(arr1, arr2, label1, label2):
    plt.plot(arr1)
    plt.plot(arr2)    
    plt.legend(labels = [label1, label2])
    plt.show()

errorX = []
errorY = []
for i in range(len(expectedPredictions)):
    errorX.append(abs(expectedPredictions[i][0] - predictions[i][0]))
    errorY.append(abs(expectedPredictions[i][1] - predictions[i][1]))
    
plotLine(errorX, errorY, 'X prediction error', 'Y prediction error')
#plotLine(errorY, 'Y prediction error')





