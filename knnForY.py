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

for offset in range(1, 25):

    (trainingData, labelXArray, labelYArray) = util.loadTrainingDataWithVelocityForY(offset)
    

        
    if onlyTrainingAndCV == True:    
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
                                                                         labelYArray,
                                                                         test_size = 0.2, 
                                                                         random_state = 0)
    else:
        X_train = trainingData
        y_train = labelYArray                                                                      
    
   
    # knn model for y
    neighY = KNeighborsRegressor(n_neighbors=n_neighbors)
    neighY.fit(np.array(X_train), np.array(y_train))
    
    if onlyTrainingAndCV == True: 
        score = neighY.score(X_test,y_test)
        cvScoresY.append(score)
#   
    if onlyTrainingAndCV == False:         
        test1 = open('inputs/test01.txt');
        test1Data =  np.loadtxt(test1, delimiter = ',');     
         
        errors = []
        NUM_INPUTS = len(test1Data) - offset
        print '**********************************************'
        for i in range(1750, 1789):
            current = test1Data[i]
            features = [current[1]]
            if i != 0:
                if util.includeVelocity == True:
                    yvel = util.calculateVelocity(test1Data, i, offset, False)
                if util.includeAcceleration == True:
                    yacc = util.calculateAcceleration(test1Data, i, offset, False)
                if util.includeHeading == True:
                    heading = util.calculateHeading(current, test1Data[i-1])
                if util.includeYDirection == True:
                    ydir = util.getDirection(current, test1Data[i-1])
            else:
                yvel = 0
                heading = 0
                yacc = 0
                ydir = 0
            if util.includeVelocity == True:
                features.append(yvel)
            if util.includeAcceleration == True:
                features.append(yacc)
            if util.includeHeading == True:
                features.append(heading)
            if util.includeYDirection == True:
                features.append(ydir)
            
            tdX = np.array(util.removeColFromRow(features, 0))
            tdY = np.array(util.removeColFromRow(features, 1))
            predY = neighY.predict(tdY)
            
            actual = test1Data[i+offset][1]
            prediction = predY[0]
            actuals.append(actual)
            predictions.append(prediction)
#            errors.append(util.error([actual], [prediction]))

if onlyTrainingAndCV == False: 
#    util.plotLines(actuals, predictions, 'Actual position', 'Predicted position')
#    util.plotGraph(actuals, predictions, 'Actual position', 'Predicted position')
    util.plotLine(actuals, 'Actual position')
    util.plotLine(predictions, 'Predicted position')
#    util.plotLine(errors, 'Error graph') 
    print len(actuals), len(predictions) 
#    print np.sum(errors)  
else:
    util.plotLine(cvScoresY, 'CV label y')


