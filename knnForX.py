# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:39:23 2016

@author: badarim
"""

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

noOfFramesToPredict = 60
n_neighbors = 10
actuals = []
predictions = []
onlyTrainingAndCV = True
cvScoresX = []
cvScoresY = []

for offset in range(1, 10):

    (trainingData, labelXArray, labelYArray) = util.loadTrainingDataWithVelocityForX(offset)
    
#    print len(trainingData), len(labelXArray)
    
    if onlyTrainingAndCV == True:
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
                                                                         labelXArray,
                                                                         test_size = 0.2, 
                                                                         random_state = 0)
    else:
        X_train = trainingData
        y_train = labelXArray

                                                                      
    # knn model for x
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
    neigh.fit(np.array(X_train), np.array(y_train))
    if onlyTrainingAndCV == True:
        score = neigh.score(X_test,y_test)
        cvScoresX.append(score)
        
    
    if onlyTrainingAndCV == False:         
        test1 = open('inputs/test01.txt');
        test1Data =  np.loadtxt(test1, delimiter = ',');     
         
        errors = []
        NUM_INPUTS = len(test1Data) - offset
        print '**********************************************'
        for i in range(1750, 1789):
            current = test1Data[i]
            features = [current[0]]
            if i != 0:
                if util.includeVelocity == True:
                    xvel = util.calculateVelocity(test1Data, i, offset, True)
                if util.includeAcceleration == True:
                    xacc = util.calculateAcceleration(test1Data, i, offset, True)
                if util.includeHeading == True:
                    heading = util.calculateHeading(current, test1Data[i-1])
                if util.includeXDirection == True:
                    xdir = util.getDirection(current, test1Data[i-1])
            else:
                xvel = 0
                heading = 0
                xacc = 0
                xdir = 0
            if util.includeVelocity == True:
                features.append(xvel)
            if util.includeAcceleration == True:
                features.append(xacc)
            if util.includeHeading == True:
                features.append(heading)
            if util.includeXDirection == True:
                features.append(xdir)
            
            tdX = np.array(util.removeColFromRow(features, 0))
            tdY = np.array(util.removeColFromRow(features, 1))
            predX = neigh.predict(tdX)
            
            actual = test1Data[i+offset][0]
            prediction = [predX[0]]
            actuals.append(actual)
            predictions.append(prediction)
            errors.append(util.error([actual], [prediction]))

if onlyTrainingAndCV == False: 
#    util.plotLines(actuals, predictions, 'Actual position', 'Predicted position')
#    util.plotGraph(actuals, predictions, 'Actual position', 'Predicted position')
    util.plotData(actuals, 'Actual position')
    util.plotData(predictions, 'Predicted position')
    util.plotLine(errors, 'Error graph') 
    print len(actuals), len(predictions) 
    print np.sum(errors)  
else:
    util.plotLine(cvScoresX, 'CV label x')


