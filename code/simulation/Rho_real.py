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

import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn import preprocessing

gamma = 0.0
def rho_real(Matrix,n,t,c,p_removes,dx,pic_rho_real_savepath):
    
    data_win = copy.deepcopy(Matrix)
    N_data_win,T_data_win = data_win.shape
#    noise_matrix = np.random.normal(0,1,size=(N_data_win,T_data_win))
#    data_win = data_win+gamma*noise_matrix
#    data_win = preprocessing.scale(data_win,axis=1)
    
    Cov_data_win = (1.0/T_data_win)*np.dot(data_win,data_win.T)
    Eig_value,Eig_vector = np.linalg.eig(Cov_data_win) 
    Eig_value = np.real(Eig_value)


    min_Eig_value = min(Eig_value)
#    print (max(Eig_value))
    if (min_Eig_value<=0):
        min_Eig_value = 0.0001
    bins = np.int((max(Eig_value)-min_Eig_value)/dx)
    counts,centers = np.histogram(Eig_value,bins)
    centers = centers[1:]
    counts_index = [index for index in range(len(counts)) if counts[index]>=0]
    counts = counts[counts_index]
    centers = centers[counts_index]
    pdf = 1.0*counts/sum(counts)/dx
    bins_real = len(centers)
    DrawMPlaw = 0
    if (DrawMPlaw):
        std = np.std(data_win)
        a = std*np.square(1-np.sqrt(c))
        if (a==0):
            a=0.0001
        b = std*np.square(1+np.sqrt(c))
        x = np.linspace(a,b)
        fx=1.0/2/np.pi/c/x/std*np.sqrt((x-a)*(b-x))
        plt.figure()
        
        plt.bar(centers,pdf,width=dx,facecolor='b',edgecolor='black')
        plt.plot(x,fx,'r')
        plt.legend(['Marchenko-Patur Law','Emperical Eigenvalue Distribution'])
        plt.xlabel('Eigenvalue')
        plt.ylabel('Probability Density')
        plt.savefig(pic_rho_real_savepath+np.str(p_removes),dpi=200)
        plt.close()
    return centers,pdf,bins_real

        
        
        
        
        
