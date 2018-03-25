# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:33:39 2016

@author: badarim
"""

# Fill in the matrices P, F, H, R and I at the bottom
#
# This question requires NO CODING, just fill in the 
# matrices where indicated. Please do not delete or modify
# any provided code OR comments. Good luck!

from math import *
from matrix import matrix
import numpy as np


print "### 4-dimensional example ###"
f = open('training_data.txt');
data =np.loadtxt(f, delimiter = ',');

initX = data[0][0]
initY = data[0][1]

trainingData = []
testData = []
numberOfPredictions = 1
splitIndex = len(data) - 1 - numberOfPredictions

for i in range(35000, len(data)):
    if( i < splitIndex):
#        trainingData.append([[data[i][0]], [data[i][1]], [0.], [0,]]);
        trainingData.append(data[i]);
    else:
        testData.append(data[i]);
        
measurements = trainingData
initial_xy = [4., 12.]

# measurements = [[1., 4.], [6., 0.], [11., -4.], [16., -8.]]
# initial_xy = [-4., 8.]

# measurements = [[1., 17.], [1., 15.], [1., 13.], [1., 11.]]
# initial_xy = [1., 19.]

dt = 0.1

x = matrix([[initial_xy[0]], [initial_xy[1]], [0.], [0.]]) # initial state (location and velocity)
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

def filter(x, P, measurements, predictions):
    for n in range(len(measurements)):
        
        # prediction
        x = (F * x) + u
        P = F * P * F.transpose()
        
        # measurement update
        Z = matrix([measurements[n]])
        y = Z.transpose() - (H * x)
        S = H * P * H.transpose() + R
        K = P * H.transpose() * S.inverse()
        x = x + (K * y)
        P = (I - (K * H)) * P
    
    print 'x= '
    x.show()
    print 'P= '
    P.show()
        
    predictions.append([x.value[0][0], x.value[1][0]])
    measurements.append([x.value[0][0], x.value[1][0]])
    return (x, P, measurements, predictions)

predictions = []
for i in range(numberOfPredictions):
    (x, P, measurements, predictions) = filter(x, P, measurements, predictions)

for d in trainingData:
    print d[0], d[1]

print testData
print predictions

# np.savetxt('testData-kf.txt', testData, delimiter=',', fmt='%10.2f');
    # np.savetxt('predictions-kf.txt', predictions, delimiter=',', fmt='%10.2f');