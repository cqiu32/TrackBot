# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:33:12 2016

@author: badarim
"""
import utilities as util
import numpy as np

for index in range(1, 11):
    f1 = open('predictions/predictions-test-' + str(index) + '.txt');
    predictions = np.loadtxt(f1, delimiter = ','); 
    
    f2 = open('predictions/actuals-test-' + str(index) + '.txt');
    actuals = np.loadtxt(f2, delimiter = ',');  
    
    util.plotGraph(actuals, predictions, 'Actual', 'Predicted')
    
    f1.close() 
    f2.close()
    
    e = open('predictions/errors-test-' + str(index) + '.txt');
    errors = np.loadtxt(e, delimiter = ',');   
    
    
    util.plotLine(errors, 'Error graph')
    e.close()
    
