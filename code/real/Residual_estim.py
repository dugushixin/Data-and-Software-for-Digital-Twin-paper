# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 21:32:53 2017

@author: Administrator
"""

import numpy as np
# from sklearn import preprocessing
# from TestFunction import TestFunction
def residual_estim(R,p_removes):
    N,T = R.shape
    Cov = 1.0/N*np.dot(R.T,R)
    Eig_value,Eig_vector = np.linalg.eig(Cov)
    Eig_value = np.real(Eig_value)
#    print(Eig_value)
#    Eig_value = 1.0/np.trace(Cov)*Eig_value_org
                
    
    Eig_value_sort = np.argsort(-Eig_value)#in default,np.argsort() sorts by ascending-
    Eig_value_select_index = Eig_value_sort[:p_removes]
#    Eig_value_select = Eig_value[Eig_value_select_index]
##        print (Eig_value_select)
#    #if (p_removes>0):
#    Eig_value_norm = Eig_value_select
##        print (Eig_value_norm)
##        Eig_value_norm = 1.0/np.sum(Eig_value)*Eig_value_select
#    les = TestFunction(fun_no,Eig_value_norm)

    Eig_vector_select = Eig_vector[:,Eig_value_select_index]
    F_estim = Eig_vector_select.T
    if (F_estim.shape[0]>0):
        L_estim = np.real(np.dot(R,np.linalg.pinv(np.mat(F_estim))))
    else:
        L_estim = np.zeros([N,p_removes],dtype='float64')
    #F_inv = np.dot(R,F_estim.T)/np.dot(F_estim,F_estim.T)
    
    U_estim = R - np.dot(L_estim,F_estim)
    U_estim = np.real(U_estim)
#    else:
#        U_estim = R
#        les = 0
#    U_estim = preprocessing.scale(U_estim,axis=1)
#    else:
#        U_estim = R
#        U_estim = preprocessing.scale(U_estim,axis=1)
#        les = 0
    return U_estim