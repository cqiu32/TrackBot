# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:02:02 2016

@author: badarim
"""


# This is just a sandbox file I'm creating to start of experimenting with different algorithms

# I'm using Knn for starters; The plan is to try EKF or others using http://filterpy.readthedocs.org/en/latest/kalman/ExtendedKalmanFilter.html

import numpy as np
import utilities as util
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation


n_neighbors = 10
offset = 1
actuals = []
predictions = []

for offset in range(1, 10):

    (trainingData, labelXArray, labelYArray) = util.loadTrainingDataWithVelocity(offset)
    
    print len(trainingData), len(labelXArray)
    
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
                                                                         labelXArray,
                                                                         test_size = 0.2, 
                                                                         random_state = 0)
    # knn model for x
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
    neigh.fit(np.array(X_train), np.array(y_train))
    
#    score = neigh.score(X_test,y_test)
#    print "CV score(x) is ", score
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
                                                                         labelYArray,
                                                                         test_size = 0.2, 
                                                                         random_state = 0)
                                                                         
    # knn model for y
    neighY = KNeighborsRegressor(n_neighbors=n_neighbors)
    neighY.fit(np.array(X_train), np.array(y_train))
    
#    score = neighY.score(X_test,y_test)
#    print "CV score(y) is ", score       
#            
    test1 = open('inputs/test01.txt');
    test1Data =  np.loadtxt(test1, delimiter = ',');     
     
    errors = []
    NUM_INPUTS = len(test1Data) - offset
    print '**********************************************'
    for i in range(NUM_INPUTS):
    #    index = ran.randint(0, len(test1Data) -1)
#        td = np.array(test1Data[i])
        current = test1Data[i]
        features = [current[0], current[1]]
        if i != 0:
            if util.includeVelocity == True:
                xvel = util.calculateVelocity(test1Data, i, offset, True)
                yvel = util.calculateVelocity(test1Data, i, offset, False)
            if util.includeAcceleration == True:
                xacc = util.calculateAcceleration(test1Data, i, offset, True)
                yacc = util.calculateAcceleration(test1Data, i, offset, False)
            if util.includeHeading == True:
                heading = util.calculateHeading(current, test1Data[i-1])
#        if i != 0:
#            xvel = util.calculateVelocity(test1Data, i, offset, True)
#            yvel = util.calculateVelocity(test1Data, i, offset, False)
#            heading = util.calculateHeading(current, test1Data[i-1])
#            xacc = util.calculateAcceleration(test1Data, i, offset, True)
#            yacc = util.calculateAcceleration(test1Data, i, offset, False)
        else:
            xvel = 0
            yvel = 0
            heading = 0
            xacc = 0
            yacc = 0
        if util.includeVelocity == True:
            features.append(xvel)
            features.append(yvel)
        if util.includeAcceleration == True:
            features.append(xacc)
            features.append(yacc)
        if util.includeHeading == True:
            features.append(heading)
        td = np.array(features)
    #    print td
    #    print '*********'
    #    # This makes it into a 2d array
    #    td = np.array(td).reshape((len(td), 1))
    #    print td
        
        predX = neigh.predict(td)
        predY = neighY.predict(td)
        
        actual = test1Data[i+offset]
        prediction = [predX[0], predY[0]]
        actuals.append(actual)
        predictions.append(prediction)
        errors.append(util.error([actual], [prediction]))

print len(actuals), len(predictions)
print np.sum(errors)
util.plotLines(actuals, predictions, 'Actual position', 'Predicted position')
util.plotGraph(actuals, predictions, 'Actual position', 'Predicted position')
util.plotLine(errors, 'Error graph')    



