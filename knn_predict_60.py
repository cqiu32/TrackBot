# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 13:08:57 2016

@author: badarim
"""

import numpy as np
import matplotlib.pyplot as plt
import math as math
import pickle as pickle
from sklearn.externals import joblib
import time

import numpy as np
#import utilities as util
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation

################## UTILITIES ####################################

NUM_FRAMES_TO_PREDICT = 60

def readFile(path):
    f =open(path)
    return np.loadtxt(f, delimiter = ",");

def getNextSteps(numberOfSteps, data, curIndex):
    stepsX = []
    stepsY = []
    for index in range(curIndex, curIndex+numberOfSteps):
        stepsX.append(data[index][0])
        stepsY.append(data[index][1])
    return (stepsX, stepsY)        

def createNextStepsMap(data):
    mapX = dict()
    mapY = dict()
    
    for i in range(len(data) - NUM_FRAMES_TO_PREDICT):
        x = data[i][0]
        y = data[i][1]
        (stepsX, stepsY) = getNextSteps(NUM_FRAMES_TO_PREDICT, data, i)
        mapX[x] = stepsX
        mapX[y] = stepsY
    return (mapX, mapY)

def createFeaturesAndLabels(data):
    features = []
    for i in range(len(data) - NUM_FRAMES_TO_PREDICT):
       current = data[i]
       (stepsX, stepsY) = getNextSteps(NUM_FRAMES_TO_PREDICT, data, i)
       if i == 0:
           features.append([current[0], current[1], 0, 0, 0, stepsX, stepsY])
       else:
           prev = data[i-1]
           velocity = getVelocity(current, prev)
           heading = calculateHeading(current, prev)
           features.append([current[0], current[1], velocity[0], velocity[1], heading, stepsX, stepsY])
    return features

def getVelocity(current, prev):
    return (30 * (current[0] - prev[0]), 30 * (current[1] - prev[1]))
    
    
def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += math.pi * 2
    return ((a + math.pi) % (math.pi * 2)) - math.pi
    
def calculateHeading(current, prev):
    heading = math.atan2(current[1] - prev[1], current[0] - prev[0])
    heading = angle_trunc(heading)
    return heading

def error(l1, l2):
    return sum((c - a)**2 + (d - b)**2 for ((a, b), (c, d)) in zip(l1, l2))**0.5  
    
def plotLine(arr1, label1):
    plt.plot(arr1)
    plt.legend(labels = [label1])
    plt.show() 

def plotGraph(arr1, arr2, label1, label2):
   plt.plot(np.array(arr1)[:,0], np.array(arr1)[:,1], 'ro')
   plt.axis([0,600, 0,600])
   
   plt.plot(np.array(arr2)[:,0], np.array(arr2)[:,1], 'bo')
   plt.legend(labels = [label1, label2])
   plt.show()     
   
def saveModel(model, path):
    # save the classifier
    with open(path + '.pkl', 'wb') as fid:
        pickle.dump(model, fid)    

def loadModel(path):
    # load the model
    with open(path + '.pkl', 'rb') as fid:
        model = pickle.load(fid)
        return model
################## UTILITIES ####################################

trainingFilePath = 'training_data.txt'
n_neighbors = 10
modelPathX = "model/knnX";
modelPathY = "model/knnY";
retrain = False

if retrain == False:
    # Load model 
    neigh  = loadModel(modelPathX)
    neighY = loadModel(modelPathY)   
else:

    data = readFile(trainingFilePath)
    #data = util.normalizeData(data)
    featuresAndLabels = createFeaturesAndLabels(data)
    
    trainingData = [ [row[0], row[1], row[2], row[3], row[4]] for row in featuresAndLabels]
    
    labelsX = [ row[5] for row in featuresAndLabels]
    labelsY = [ row[6] for row in featuresAndLabels]
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
                                                                             labelsX,
                                                                             test_size = 0.2, 
                                                                             random_state = 0)
    
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
    neigh.fit(np.array(X_train), np.array(y_train))
    
    score = neigh.score(X_test,y_test)
    print "CV (X)", score
        
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData,
                                                                         labelsY,
                                                                         test_size = 0.2, 
                                                                         random_state = 0)
    # knn model for y
    neighY = KNeighborsRegressor(n_neighbors=n_neighbors)
    neighY.fit(np.array(X_train), np.array(y_train))
    
    score = neighY.score(X_test,y_test)
    print "CV (y)", score 
     
    # Save the model
    saveModel(neigh, modelPathX)
    saveModel(neighY, modelPathY)                                                                          

import os
errors = []
for filename in os.listdir(os.getcwd()+'/inputs'):   
    print filename
#    testFilePath = 'inputs/test01.txt'
    testFilePath = 'inputs/' + filename
    testdata = readFile(testFilePath)
    
    # LAST_ROW_INDEX = len(testdata) - 1
    LAST_ROW_INDEX = len(testdata) - 1 - NUM_FRAMES_TO_PREDICT # this is just for testing
    
    testFeaturesAndLabels = createFeaturesAndLabels(testdata)
    
    testFeatureData = [ [row[0], row[1], row[2], row[3], row[4]] for row in testFeaturesAndLabels]
    
    testActualLabelsX = [ row[5] for row in testFeaturesAndLabels]
    testActualLabelsY = [ row[6] for row in testFeaturesAndLabels]
    
    lastRowFeature = np.array(testFeatureData[LAST_ROW_INDEX])
    lastRowFeature = lastRowFeature.reshape(1, -1)
    lasRowLabelX = testActualLabelsX[LAST_ROW_INDEX]
    lasRowLabelY = testActualLabelsY[LAST_ROW_INDEX]   
        
    predictionsX = neigh.predict(lastRowFeature)[0]
    predictionsY = neighY.predict(lastRowFeature)[0]     
    
    predictions = zip(predictionsX, predictionsY)
    actuals = zip(lasRowLabelX, lasRowLabelY)    
    
    
    plotGraph(actuals, predictions, 'Actual position', 'Predicted position')
        
    errors.append(error(actuals, predictions))
    
plotLine(errors, 'Error by frame #') 
errors = sorted(errors)
print sum(errors[1:-1]) / (len(errors) - 2)
    