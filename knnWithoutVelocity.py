# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 00:34:40 2016

@author: badarim
"""

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

n_neighbors = 5

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
#        xvel = 30 * (data[i][0] - prev[0])
#        yvel = 30 * (data[i][1] - prev[1])
    trainingData.append([val[0], val[1]])
    labelsX = []
    labelsY = []
    for j in range(1, noOfFramesToPredict+1):
        # get next 60 frames of x, y positions as labels for regression.
        labelsX.append(data[j][0])
        labelsY.append(data[j][1])        
    labelXArray.append(labelsX)
    labelYArray.append(labelsY)

print len(trainingData), len(labelXArray), len(labelYArray)


# create an array for every frame
modelsX = []
for i in range(noOfFramesToPredict):
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
    temp = [yt[i] for yt in labelXArray]
    neigh.fit(trainingData, temp)
    modelsX.append(neigh)


modelsY = []
for i in range(noOfFramesToPredict):
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
    
    temp = [yt[i] for yt in labelYArray]
    neigh.fit(trainingData, temp)
    modelsY.append(neigh)                                                                         
        
test1 = open('inputs/test01.txt');
test1Data =  np.loadtxt(test1, delimiter = ',');       
errors = []
NUM_INPUTS = 1
for i in range(NUM_INPUTS):
    index = ran.randint(0, len(test1Data) -1)
    features = []
    if index != 0:
        velX = 30 * (test1Data[i][0] - test1Data[i-1][0])
        velY = 30 * (test1Data[i][1] - test1Data[i-1][1])
        features = [test1Data[i][0], test1Data[i][1]]
        predictions = []
        predictions = []
        for m in range(len(modelsX)):
            modX = modelsX[m]
            modY = modelsY[m]
            predictions.append([modX.predict(features), modY.predict(features)])
        print "For row: ", test1Data[i]
        ctr = 1
        for row in predictions:
            print row , " : " , test1Data[ctr]
            ctr = ctr + 1
        np.savetxt('predictions-knn-60.txt', predictions, delimiter=',', fmt='%10.3f');








