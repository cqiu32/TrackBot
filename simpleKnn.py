# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:02:02 2016

@author: poornima
"""

import numpy as np
import utilities as util
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation

noOfFramesToPredict = 60
n_neighbors = 10
actuals = []
predictions = []
cvScoresX = []
cvScoresY = []
offset = 1
retrain = False
runPredictions = True
modelPathX = "model/x/knnX";
modelPathY = "model/y/knnY";


if retrain == False:
    # Load model 
    modelX = util.loadModel(modelPathX)
    modelY = util.loadModel(modelPathY)   
else:
    ####################################### TRAINING #######################################
    (trainingData, labelXArray, labelYArray) = util.loadTrainingDataWithVelocity(offset)
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
                                                                         labelXArray,
                                                                         test_size = 0.2, 
                                                                         random_state = 0)
    # knn model for x
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
    neigh.fit(np.array(X_train), np.array(y_train))
    
    score = neigh.score(X_test,y_test)
    cvScoresX.append(score)
    print "CV (X)", score
        
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
                                                                         labelYArray,
                                                                         test_size = 0.2, 
                                                                         random_state = 0)
    # knn model for y
    neighY = KNeighborsRegressor(n_neighbors=n_neighbors)
    neighY.fit(np.array(X_train), np.array(y_train))
    
    score = neighY.score(X_test,y_test)
    cvScoresY.append(score)
    print "CV (y)", score
     
    # Save the model
    util.saveModel(neigh, modelPathX)
    util.saveModel(neighY, modelPathY)   
        
         
####################################### TESTING #######################################
if runPredictions == True:
    for testIndex in range(1, 11):
        if testIndex == 10:
            test1 = open('inputs/test10.txt');
        else:
            test1 = open('inputs/test0' + str(testIndex) + '.txt');
        test1Data =  np.loadtxt(test1, delimiter = ',');     
        errors = []
        length = len(test1Data)
        for i in range(length - noOfFramesToPredict-offset, length-offset):
            current = test1Data[i]
    
            testRow = util.getClosestRowFromTraining(current, trainingData)        
            features = util.createFeatureRow(test1Data, i, offset, testRow)
            
            td = np.array(features)
            
            predX = modelX.predict(td)
            predY = modelY.predict(td)
            
            actual = test1Data[i+offset]
            prediction = [predX[0], predY[0]]
    
            actuals.append(actual)
            predictions.append(prediction)
            errors.append(util.error([actual], [prediction]))
            
#        np.savetxt('actual/' + str(testIndex) + '.txt', actuals, delimiter=',', fmt='%d');
#        np.savetxt('predictions/predictions-test-' + str(testIndex) + '.txt', predictions, delimiter=',', fmt='%d');
#        np.savetxt('predictions/errors-test-' + str(testIndex) + '.txt', errors, delimiter=',', fmt='%d');

    
    
        util.plotGraph(actuals, predictions, 'Actual position', 'Predicted position')
        util.plotLine(errors, 'Error graph') 
        