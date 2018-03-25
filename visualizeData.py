# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:00:14 2016

@author: badarim
"""

import numpy as np
import utilities as util

t = open('inputs/test01.txt');
f = open('training_data.txt');
data = np.loadtxt(f, delimiter = ','); 

util.plotData(data, 'Training data')
data = np.loadtxt(t, delimiter = ','); 
util.plotData(data, 'Test1 data')