# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 22:03:51 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:08:14 2017

@author: shixin
"""
#import math
#import get_file_list
#import data_load
#import data_preprocessing
#import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('agg')
#matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
import copy
from sklearn import preprocessing
#import TestFunction
pic_mp_path = 'mp/'
gamma = 0.2
def rho_real(Matrix,n,t,c,step_k,p_removes):
#    LES = []
                         
#    for step_k in range(step_num):
#    start = step*step_k
#    end = t+step*step_k
    data_win = copy.deepcopy(Matrix)
    #parameter set
#        N_data_win = data_win.shape[0]
    N_data_win,T_data_win = data_win.shape
#    noise_matrix = np.random.normal(0,1,size=(N_data_win,T_data_win))
#    data_win = data_win+gamma*noise_matrix
    data_win = np.real(data_win)
    data_win = preprocessing.scale(data_win,axis=1)
#    
    Cov_data_win = (1.0/T_data_win)*np.dot(data_win,data_win.T)
    Eig_value,Eig_vector = np.linalg.eig(Cov_data_win) 
    Eig_value = np.real(Eig_value)
#    print (max(Eig_value))
#    print (Eig_value)
#    print (max(Eig_value))
#        fun_no = 'topfew'
#        fun_no = 'entropy'
#        fun_no = 'entropy'
#        les = TestFunction.TestFunction(fun_no,Eig_value,Eig_vector)
#        LES.append(les)
    dx = 0.15
    min_Eig_value = min(Eig_value)
    print (max(Eig_value))
    if (min_Eig_value<=0):
        min_Eig_value = 0.0001
    bins = np.int((max(Eig_value)-min_Eig_value)/dx)
    counts,centers = np.histogram(Eig_value,bins)
    centers = centers[1:]
    counts_index = [index for index in range(len(counts)) if counts[index]>0]
    counts = counts[counts_index]
    centers = centers[counts_index]
    pdf = 1.0*counts/sum(counts)/dx
    bins_real = len(centers)
    DrawMPlaw = 0
    if (DrawMPlaw):
        var = np.var(data_win)
        a = var*np.square(1-np.sqrt(c))
        if (a==0):
            a=0.0001
        b = var*np.square(1+np.sqrt(c))
        x = np.linspace(a,b)
        fx=1.0/2/np.pi/c/x/var*np.sqrt((x-a)*(b-x))
        plt.figure()
        
        plt.bar(centers,pdf,width=dx,facecolor='b')
        plt.plot(x,fx,'r')
        plt.legend(['Marchenko-Pastur律','经验特征值分布'])
        plt.xlabel('特征值')
        plt.ylabel('概率密度')
        plt.savefig(pic_mp_path+np.str(step_k)+'_'+np.str(p_removes),dpi=300)
        plt.show()
        plt.close()
    return centers,pdf,bins_real

        
        
        
        
        