# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 21:32:53 2017

@author: Administrator
"""

import numpy as np

def residual_estim(R,p_removes):
    N,T = R.shape
    Cov = 1.0/N*np.dot(R.T,R)
    Eig_value_org,Eig_vector = np.linalg.eig(Cov)
    Eig_value = Eig_value_org
    
    Eig_value_sort = np.argsort(-Eig_value)#in default,np.argsort() sorts by ascending-
    Eig_value_select_index = Eig_value_sort[:p_removes]
    if (p_removes>0):
        Eig_vector_select = Eig_vector[:,Eig_value_select_index]
        F_estim = Eig_vector_select.T
        if (F_estim.shape[0]>0):
            L_estim = np.real(np.dot(R,np.linalg.pinv(np.mat(F_estim))))
        else:
            L_estim = np.zeros([N,p_removes],dtype='float64')
        U_estim = R - np.dot(L_estim,F_estim)
    else:
        U_estim = R

    return U_estim