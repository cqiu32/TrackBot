# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 19:54:34 2016

@author: badarim
"""
from filterpy.kalman import KalmanFilter
import numpy as np
from filterpy.common import Q_discrete_white_noise

NumberOfPredictions = 60
f = open('training_data.txt');
data =np.loadtxt(f, delimiter = ',');

initX = data[0][0]
initY = data[0][1]

trainingData = []
testData = []
splitIndex = 35000

for i in range(1, len(data)):
    if( i < splitIndex):
#        trainingData.append([[data[i][0]], [data[i][1]], [0.], [0,]]);
        trainingData.append(data[i]);
    else:
        testData.append(data[i]);
    
    
noOfStateVariables = 4
noOfMeasurementInputs = len(trainingData)
kf = KalmanFilter (dim_x=noOfStateVariables, dim_z=2)


#f.x = np.array([[2.],    # position
#                [0.]])   # velocity
kf.x = np.array([[initX], [initY], [0.], [0.]])
T = 0.034
kf.F = np.array([[1.,0., T, 0],
                [0.,1., 0, T],
                [0.,0.,1.,0.],
                [0.,0.,0.,1.]
              ])

kf.R = np.array([[1, 0],
               [0, 1]])
kf.H = np.array([[1.,0.,0.,0],
                [0.,1.,0.,0]])               
kf.P = np.array([[0.,0.,0.,0],
                [0.,0.,0.,0],
                [0.,0.,1000.,0],
                [0.,0.,0.,1000.]])

#f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

#while True:
#    z, R = read_sensor()
#    x, P = predict(x, P, F, Q)
#    x, P = update(x, P, z, R, H)

(mu, cov, _, _) = kf.batch_filter(trainingData, update_first=False)
(x, P, K) = kf.rts_smoother(mu, cov, kf.F, kf.Q)
#x = [initX, initY]    
#for i, z in enumerate(trainingData):
#    prior = f.predict(x)
#    f.update(z, f.R, f.H)

