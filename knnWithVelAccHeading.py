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
import pickle

noOfFramesToPredict = 60
n_neighbors = 7
actuals = []
predictions = []
onlyTrainingAndCV = True
cvScoresX = []
cvScoresY = []

retrain = onlyTrainingAndCV
modelPath = 'models/'


for offset in range(1, 25):
    if retrain == True:
        (trainingData, labelXArray, labelYArray) = util.loadTrainingDataWithVelocity(offset)
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
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors,algorithm='kd_tree')
        neigh.fit(np.array(X_train), np.array(y_train))
        if onlyTrainingAndCV == True:
            score = neigh.score(X_test,y_test)
            cvScoresX.append(score)
        
#        util.saveModel(neigh, modelPath + 'knnX' + str(offset))
        if onlyTrainingAndCV == True:    
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
                                                                             labelYArray,
                                                                             test_size = 0.2, 
                                                                             random_state = 0)
        else:
            X_train = trainingData
            y_train = labelYArray                                                                      
        
        # knn model for y
        neighY = KNeighborsRegressor(n_neighbors=n_neighbors,algorithm='kd_tree')
        neighY.fit(np.array(X_train), np.array(y_train))
        
        if onlyTrainingAndCV == True: 
            score = neighY.score(X_test,y_test)
            cvScoresY.append(score)
#        util.saveModel(neighY, modelPath + 'knnY' + str(offset))
#    else:
#        neigh = util.loadModel(modelPath + 'knnX' + str(offset))
#        neighY = util.loadModel(modelPath + 'knnY' + str(offset))
        
    if onlyTrainingAndCV == False:         
        test1 = open('inputs/test01.txt');
        test1Data =  np.loadtxt(test1, delimiter = ',');     
        test1Data = util.normalizeData(test1Data)
        errors = []
        NUM_INPUTS = len(test1Data) - offset
        for i in range(1750, NUM_INPUTS):
            current = test1Data[i]
            features = util.createFeatureRow(test1Data, i, offset, current)
            td = np.array(features)
            predX = neigh.predict(td)
            predY = neighY.predict(td)
            
            actual = test1Data[i+offset]
            prediction = [predX[0], predY[0]]
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
    util.plotLine(cvScoresY, 'CV label y')
#    util.plotLines(cvScoresX, cvScoresY, 'CV label X', 'CV label y')

