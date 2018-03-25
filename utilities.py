# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 19:48:36 2016

@author: badarim
"""
import numpy as np
import matplotlib.pyplot as plt
import math as math
import pickle as pickle
from sklearn.externals import joblib
import matplotlib.animation as animation
import time

includeAcceleration = False
includeVelocity = True
includeHeading = True
includeTurningAngle = False 
includeYDirection = False
includeXDirection = False
includeDistance = True
includeCollisionFlag = False
useBothPositions = True
useAverageVelocity = True
normalizeFlag = False

def loadTrainingData(offset):
#    f = open('training_data.txt');
    f = open('inputs/test01.txt');
    data = np.loadtxt(f, delimiter = ',');    
    td = []
    for n in range(len(data) - offset):
        td.append(data[n])
    labelXArray = []
    labelYArray = []
    for i in range(len(data)):
        if(i+offset < len(data)):
            val = data[i+offset]
            labelXArray.append(val[0]);
            labelYArray.append(val[1]);
    return (td, labelXArray, labelYArray)

def saveModel(model, path):
# save the classifier
    with open(path + '.pkl', 'wb') as fid:
        pickle.dump(model, fid)    

    # load it again

#    joblib.dump(model, path + '.pkl') 
#    with open(path, 'wb') as f:
#        pickle.dump(object, f)    

def loadModel(path):
    with open(path + '.pkl', 'rb') as fid:
        model = pickle.load(fid)
        return model
#    joblib.load(path + '.pkl')
#    with open(path, 'rb') as f:
#        return pickle.load(f)
    
def calculateAcceleration(data, currentIndex, offset, forX = True):
    def basicVel(i):
        current = data[i]
        prev = data[i-1]
        if forX == True:
            return 30 * (current[0] - prev[0])
        else:
            return 30 * (current[1] - prev[1])
    numOfPositionsForAvgVelocity = 5
    velocities = []
    index = currentIndex
    ctr = 0
    while index > 0 and ctr <= numOfPositionsForAvgVelocity:
        velocities.append(basicVel(index))
        index = index - 1
        ctr = ctr + 1
    
    numOfComb = len(velocities) - 1   
    if numOfComb == 0:
        return 0
    index = numOfComb
    acc = 0
    while index >= 1:
        acc += (velocities[index] - velocities[index-1]) * 30
        index = index -1
    
    return acc/(len(velocities) - 1)

def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += math.pi * 2
    return ((a + math.pi) % (math.pi * 2)) - math.pi
    
def calculateHeading(current, prev):
    heading = math.atan2(current[1] - prev[1], current[0] - prev[0])
    heading = angle_trunc(heading)
    return heading
    
def calculateVelocity(data, currentIndex, offset, forX = True):
    def basicVel(i):
        current = data[i]
        prev = data[i-1]
        if forX == True:
            return 30 * (current[0] - prev[0])
        else:
            return 30 * (current[1] - prev[1])
    if useAverageVelocity:
        numOfPositionsForAvgVelocity = 5
        velSum = 0
        index = currentIndex
        ctr = 0
        while index > 0 and ctr <= numOfPositionsForAvgVelocity:
            velSum += basicVel(index)
            index = index - 1
            ctr = ctr + 1
        return velSum/ctr
    else:
        return basicVel(currentIndex)

def getDirection(current, prev, forX=True):
    if forX == True:
        if current[0] < prev[0]:
            return -50;
        else:
            return 50;
    else:
        if current[1] < prev[1]:
            return -50;
        else:
            return 50;
    
def contains(array, d):
    for t in array:
        if t == d:
            return True
    return False

def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  
def clean_cos(cos_angle):
    return min(1,max(cos_angle,-1))
    
def getTurningAngle(data, i):
    if i <= 2:
        return 0;
    cur = data[i]
    curMinus1 = data[i-1]
    curMinus2 = data[i-2]
    
    deltaX1 = curMinus1[0] - curMinus2[0]
    deltaY1 = curMinus1[1] - curMinus2[1]
    dist1 = distance_between(curMinus1, curMinus2)
    
    deltaX2 = cur[0] - curMinus1[0]
    deltaY2 = cur[1] - curMinus1[1]
    dist2 = distance_between(cur, curMinus1)
    
    return math.acos(clean_cos((deltaX1 * deltaX2 + deltaY1 * deltaY2)/(dist1 * dist2)))
    
def removeColFromRow(row, colsIndexToRemove):
    if useBothPositions == True:
        return row
    newRow = []
    for i in range(len(row)):            
        if contains(colsIndexToRemove, i) == False:
            newRow.append(row[i])
    return newRow

def removeColFromMatrix(data, colsIndexToRemove):
    newData = []
    for row in data:
        newData.append(removeColFromRow(row, colsIndexToRemove))
    return newData

def collided(current):
    x = current[0]
    y = current[1]
    if x < 68 or x > 496 or y < 34 or y > 293:
        return -50
    else:
        return 50
    
def getClosestRowFromTraining(current, trainingData):
    closestRow = [0,0]
    for d in trainingData:
        temp = [d[0], d[1]]
        if distance_between(current, temp) < distance_between(closestRow, current):
            closestRow = temp;
    return closestRow
    
def createFeatureRow(data, i, offset, current):
    features = [current[0], current[1]]
    if i != 0:
        if includeVelocity == True:
            xvel = calculateVelocity(data, i, offset, True)
            yvel = calculateVelocity(data, i, offset, False)
        if includeAcceleration == True:
            xacc = calculateAcceleration(data, i, offset, True)
            yacc = calculateAcceleration(data, i, offset, False)
        if includeHeading == True:
                heading = calculateHeading(current, data[i-1])
#            if i+offset < len(data):
#                heading = calculateHeading(data[i+offset], current)
#            else:
#                heading = 0
        if includeXDirection == True:
            xdir = getDirection(current, data[i-1])
        if includeYDirection == True:
            ydir = getDirection(current, data[i-1])
        if includeTurningAngle == True:
            ta = getTurningAngle(data, i)
        if includeCollisionFlag == True:
            coll = collided(current)
        if includeDistance == True:
            dist = distance_between(current, data[i-1])
    else:
        xvel = 0
        yvel = 0
        heading = 0
        xacc = 0
        yacc = 0
        xdir = 0
        ydir = 0
        ta = 0
        coll = 0
        dist = 0
    if includeVelocity == True:
        features.append(xvel)
        features.append(yvel)
    if includeAcceleration == True:
        features.append(xacc)
        features.append(yacc)
    if includeHeading == True:
        features.append(heading)
    if includeXDirection == True:
        features.append(xdir)
    if includeYDirection == True:
        features.append(ydir)
    if includeTurningAngle == True:
        features.append(ta)
    if includeCollisionFlag == True:
        features.append(coll)
    if includeDistance == True:
        features.append(dist)
    return features
    
def loadTrainingDataWithVelocity(offset):
    f = open('training_data.txt');
    trainingData = []
    data = np.loadtxt(f, delimiter = ','); 
    
    data = normalizeData(data)
    for i in range(len(data) - offset):
        current = data[i]
#        features = [current[0], current[1]]
#        if i != 0:
#            if includeVelocity == True:
#                xvel = calculateVelocity(data, i, offset, True)
#                yvel = calculateVelocity(data, i, offset, False)
#            if includeAcceleration == True:
#                xacc = calculateAcceleration(data, i, offset, True)
#                yacc = calculateAcceleration(data, i, offset, False)
#            if includeHeading == True:
##                heading = calculateHeading(current, data[i-1])
#                heading = calculateHeading(data[i+offset], current)
#            if includeXDirection == True:
#                xdir = getDirection(current, data[i-1])
#            if includeYDirection == True:
#                ydir = getDirection(current, data[i-1])
#        else:
#            xvel = 0
#            yvel = 0
#            heading = 0
#            xacc = 0
#            yacc = 0
#            xdir = 0
#            ydir = 0
#        if includeVelocity == True:
#            features.append(xvel)
#            features.append(yvel)
#        if includeAcceleration == True:
#            features.append(xacc)
#            features.append(yacc)
#        if includeHeading == True:
#            features.append(heading)
#        if includeXDirection == True:
#            features.append(xdir)
#        if includeYDirection == True:
#            features.append(ydir)
        features = createFeatureRow(data, i, offset, current)
        trainingData.append(features)
    
    labelXArray = []
    labelYArray = []
    for i in range(len(data)):
        if(i+offset < len(data)):
            val = data[i+offset]
            labelXArray.append(val[0]);
            labelYArray.append(val[1]);
    return (trainingData, labelXArray, labelYArray)

def normalize(arr, minX, minY, dX, dY):
    return [float((arr[0] - minX)/dX), float((arr[1] - minY)/dY)]

def normalizeData(data):
    if normalizeFlag == False:
        return data
    minX = np.amin(data[:, 0])
    minY = np.amin(data[:, 1])
    dX = np.amax(data[:, 0]) - minX
    dY = np.amax(data[:, 1]) - minY
    
#    print minX, minY, dX, dY
    
    normalizedData = []
    for d in data:
        normalizedData.append(normalize(d, minX, minY, dX, dY))
    return normalizedData

def loadTrainingDataWithVelocityForY(offset):
    f = open('training_data.txt');
    trainingData = []
    data = np.loadtxt(f, delimiter = ','); 
    for i in range(len(data) - offset):
        current = data[i]
        features = [current[1]]
        if i != 0:
            if includeVelocity == True:
                yvel = calculateVelocity(data, i, offset, False)
            if includeAcceleration == True:
                yacc = calculateAcceleration(data, i, offset, False)
            if includeHeading == True:
                heading = calculateHeading(current, data[i-1])
            if includeYDirection == True:
                ydir = getDirection(current, data[i-1])
        else:
            yvel = 0
            heading = 0
            yacc = 0
            ydir = 0
        if includeVelocity == True:
            features.append(yvel)
        if includeAcceleration == True:
            features.append(yacc)
        if includeHeading == True:
            features.append(heading)
        if includeYDirection == True:
            features.append(ydir)
        trainingData.append(features)
    
    labelXArray = []
    labelYArray = []
    for i in range(len(data)):
        if(i+offset < len(data)):
            val = data[i+offset]
            labelXArray.append(val[0]);
            labelYArray.append(val[1]);
    return (trainingData, labelXArray, labelYArray)
    
def loadTrainingDataWithVelocityForX(offset):
    f = open('training_data.txt');
    trainingData = []
    data = np.loadtxt(f, delimiter = ','); 
    for i in range(len(data) - offset):
        current = data[i]
        features = [current[0]]
        if i != 0:
            if includeVelocity == True:
                xvel = calculateVelocity(data, i, offset, True)
            if includeAcceleration == True:
                xacc = calculateAcceleration(data, i, offset, True)
            if includeHeading == True:
                heading = calculateHeading(current, data[i-1])
            if includeXDirection == True:
                xdir = getDirection(current, data[i-1])
        else:
            xvel = 0
            heading = 0
            xacc = 0
            xdir = 0
        if includeVelocity == True:
            features.append(xvel)
        if includeAcceleration == True:
            features.append(xacc)
        if includeHeading == True:
            features.append(heading)
        if includeXDirection == True:
            features.append(xdir)
        trainingData.append(features)
    
    labelXArray = []
    labelYArray = []
    for i in range(len(data)):
        if(i+offset < len(data)):
            val = data[i+offset]
            labelXArray.append(val[0]);
            labelYArray.append(val[1]);
    return (trainingData, labelXArray, labelYArray)

def getLabels(data):
    labelXArray = []
    labelYArray = []
    for i in range(len(data)):
        val = data[i]
        labelXArray.append(val[0]);
        labelYArray.append(val[1]);
    return (labelXArray, labelYArray)

    
def error(l1, l2):
    return sum((c - a)**2 + (d - b)**2 for ((a, b), (c, d)) in zip(l1, l2))**0.5    

def plotGraph(arr1, arr2, label1, label2):
   plt.plot(np.array(arr1)[:,0], np.array(arr1)[:,1], 'ro')
   plt.axis([0,600, 0,600])
   
   plt.plot(np.array(arr2)[:,0], np.array(arr2)[:,1], 'bo')
   plt.legend(labels = [label1, label2])
   plt.show()     

def plotData(arr, label):
    plt.plot(np.array(arr)[:,0], np.array(arr)[:,1], 'ro')
#    plt.axis([70,570, 10,350])
   
#    plt.legend(labels = [label])
    plt.title = label
    plt.show()    
    
def plotLines(arr1, arr2, label1, label2):
    plt.plot(arr1, arr2)
#    plt.plot(arr2)    
    plt.legend(labels = [label1, label2])
    plt.show()

def plotLine(arr1, label1):
    plt.plot(arr1)
    plt.legend(labels = [label1])
    plt.show()    
    
    
def animate(dataArray):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.clear()
    ax1.plot(dataArray[:, 0],dataArray[:, 1])
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
