# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:56:15 2016

@author: poornima
"""
from math import *
from matrix import matrix
import numpy as np
import utilities as util

numberOfPredictions = 60
dt = 0.1

x = matrix([[0.], [0.], [0.], [0.]]) # initial state (location and velocity)
u = matrix([[0.], [0.], [0.], [0.]]) # external motion

#### DO NOT MODIFY ANYTHING ABOVE HERE ####
#### fill this in, remember to use the matrix() function!: ####

# initial uncertainty: 0 for positions x and y, 1000 for the two velocities
#P = matrix([[0], [0],[1000],[1000]])
P = matrix([[0,0,0,0],[0,0,0,0],[0,0,1000,0],[0,0,0,1000]])
# next state function: generalize the 2d version to 4d
F = matrix([[1,0,0.1,0], [0,1,0,0.1],[0,0,1,0],[0,0,0,1]])
# measurement function: reflect the fact that we observe x and y but not the two velocities
H = matrix([[1,0,0,0],[0,1,0,0]]) 
# measurement uncertainty: use 2x2 matrix with 0.1 as main diagonal
R = matrix([[0.1, 0],[0,0.1]])
# 4d identity matrix
I =  matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
###### DO NOT MODIFY ANYTHING HERE #######

predictions = []
actuals=[]
for i in range (1,10) :    
    f = open('inputs/test0' + str(i) + '.txt');
    testdata =np.loadtxt(f, delimiter = ',');
    errors=[]
    tempdiff=[]
    for i in range(len(testdata)-numberOfPredictions,len(testdata)):
        temp = util.getClosestRowFromTraining(testdata[i], trainingData)
        # prediction
        x = (F * x) + u
        P = F * P * F.transpose()
        
        # measurement update
        Z = matrix([temp])
        y = Z.transpose() - (H * x)
        S = H * P * H.transpose() + R
        K = P * H.transpose() * S.inverse()
        x = x + (K * y)
        P = (I - (K * H)) * P
        
        actuals.append(testdata[i])
        predictions.append([x.value[0],x.value[1]])
        errors.append(util.error([testdata[i]], [prediction]))
        tempdiff.append(util.error([temp], [prediction]))
        
    #util.plotData(actuals, 'Actual position')
    #util.plotData(predictions, 'Predicted position')
    util.plotLine(errors, 'Error graph') 
    
    #util.plotLine(tempdiff, 'tempdiff graph') 
    util.plotGraph(actuals, predictions, 'Actual position', 'Predicted position')

