# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 18:03:46 2017

@author: shixin
"""

import pandas as pd
import data_preprocessing1
import get_residual1


day_analy = int(96*2)
data_org = pd.read_csv('200534176.csv')
data = data_org.iloc[:96,1:day_analy]
data = data_preprocessing1.data_preprocessing(data)
temp = []
for i in range(data.shape[0]):
    temp_residual = get_residual1.get_residual(data[i,:])
    temp.append(temp_residual[-day_analy+5:])
    
    
import numpy as np  
import MPlaw
import Ringlaw  
Matrix = np.mat(temp)
n,t = Matrix.shape
c = 1.0*n/t
step = t
step_num = 1


#Matrix = Matrix[:,:]+0.01*np.random.normal(0,1,size=[n,t])
from sklearn import preprocessing
Matrix = preprocessing.scale(Matrix,axis=1)

data_mp_savepath = 'E:/OneDrive/temp/pic/mp_data'
residual_mp_savepath = 'E:/OneDrive/temp/pic/mp_residual'
data_LES = MPlaw.MPlaw(data,n,t,c,step,step_num,data_mp_savepath)
residual_LES = MPlaw.MPlaw(Matrix,n,t,c,step,step_num,residual_mp_savepath)

data_ring_savepath = 'E:/OneDrive/temp/pic/ring_data'
residual_ring_savepath = 'E:/OneDrive/temp/pic/ring_residual'
data_MSR = Ringlaw.Ringlaw(data,n,t,c,step,step_num,data_ring_savepath)
residual_MSR = Ringlaw.Ringlaw(Matrix,n,t,c,step,step_num,residual_ring_savepath)

