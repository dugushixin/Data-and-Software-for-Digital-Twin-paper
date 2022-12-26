# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 21:57:28 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 20:58:34 2017

@author: Administrator
"""

import numpy as np
#import math
import copy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
from sklearn import preprocessing
import Residual_estim
import Rho_real
import Rho_model


pic_eff_savepath = 'eff/'
pic_rho_real_savepath = 'rho_real/'
n = 500
p = 3#number of factors
t = n
c = 1.0*n/t

J = int(n/10)
alpha = 0.5
beta = 0.05

e_corr = np.sqrt(1.0*(1-(alpha**2))/(1+2*J*(beta**2)))

dx = 0.15
num_model = 100
best_p_removes_iters = []
best_phi1_iters = []
best_phi2_iters = []
best_distance_iters = []
mp_distance_iters_p = []
Isdraw = 1
iters = 1
SNR_list = {100}
#{1,10,100,1000,10000}
best_p_removes = []
best_phi1 = []
best_phi2 = []
for SNR in SNR_list:
    gamma = np.sqrt(1.0/SNR*p)
    for iter_num in range(iters):
        L = np.random.normal(0,1,size=[n,p])
        F = np.random.normal(0,1,size=[p,t])
        v = np.random.normal(0,1,size=[n,t])
        
        e = np.zeros([n,t],dtype=np.float64)
        for i in range(n):
            e[i,0] = v[i,0]
            for j in range(1,t):
                cross1 = 0
                cross2 = 0
                for ii in range(max(i-J,0),i-1):
                    cross1 = cross1 + beta*v[ii,j]
                for iii in range(i+1,min(i+J,n)):
                    cross2 = cross2 + beta*v[iii,j]
                e[i,j] = alpha*e[i,j-1]+v[i,j]+cross1+cross2
        U = e_corr*e        
        R = np.dot(L,F)+gamma*U
np.savetxt('05_005.csv', R,delimiter=',',fmt='%f')
'''        
        #draw the generated data
        
        for row_num in range(n):
            plt.plot(range(t),R[row_num,:],'*-')
        plt.savefig('data_org.png',dpi=200)
        
        data = copy.deepcopy(R)
        best_p_removes_iter = 0
        best_phi1_iter = 0
        best_phi2_iter = 0
        best_distance_iter = 100
        for p_removes in range(2,10):
            U_estim = Residual_estim.residual_estim(data,p_removes)#normalized data is returned
            U_estim = preprocessing.scale(U_estim,axis=1)
            centers_real,pdf_real,bins_real = Rho_real.rho_real(U_estim,n,t,c,p_removes,dx,pic_rho_real_savepath)
            min_distance_JS,opt_pdf_model,opt_phi1,opt_phi2 = Rho_model.rho_model(num_model,bins_real,centers_real,pdf_real)
            # min_distance_KL = distance_KL[0,min_index]
            # opt_pdf_model = pdf_model[min_index]
            # opt_phi = phi_list[min_index]
            if (min_distance_JS<best_distance_iter):
                best_distance_iter = min_distance_JS
                best_p_removes_iter = p_removes
                best_phi1_iter = opt_phi1
                best_phi2_iter = opt_phi2
            if (Isdraw):
                std = np.std(U_estim)
                a = std*np.square(1-np.sqrt(c))
                b = std*np.square(1+np.sqrt(c))
                x = np.linspace(a,b)
                fx=1.0/2/np.pi/c/x/std*np.sqrt((x-a)*(b-x))
                #计算mp密度函数与实际谱之间的距离
                pdf_mp = []
                for centers_real_i in centers_real:
                    temp = 0.00001
                    if a<=centers_real_i<=b:
                        temp = 1.0/2/np.pi/c/centers_real_i/std*np.sqrt((centers_real_i-a)*(b-centers_real_i))
                    pdf_mp.append(temp)
                pdf_mp = np.array(pdf_mp)
                av = 0.5*(pdf_real+pdf_mp)
                mp_distance = 0.5*np.sum(pdf_real*np.log(1.0*pdf_real/av))+0.5*np.sum(pdf_mp*np.log(1.0*pdf_mp/av))#JS distance
                # print (mp_distance)
                mp_distance_iters_p.append(mp_distance)
                plt.figure()
                plt.bar(centers_real,pdf_real,width=dx,facecolor='y',edgecolor='black')#facecolor='b',
                plt.plot(x,fx,'r')
                plt.plot(centers_real,opt_pdf_model,'k--')
                plt.tick_params(labelsize=14)
                plt.legend(['Marchenko-Pastur律','模型(p='+np.str(3)+','r'$\varphi_1$='+np.str(round(opt_phi1,4))+','r'$\varphi_2$='+np.str(round(opt_phi2,4))+')','仿真数据'],fontsize=14)
                plt.xlabel('特征值',fontsize=15)
                plt.ylabel('概率密度函数'r'$\rho$',fontsize=15)
                plt.savefig(pic_eff_savepath+np.str(p_removes),dpi=300)
                # plt.savefig(pic_eff_savepath+np.str(p_removes)+'.eps')
                plt.close()
        best_distance_iters.append(best_distance_iter)
        best_p_removes_iters.append(best_p_removes_iter)
        best_phi1_iters.append(best_phi1_iter)
        best_phi2_iters.append(best_phi2_iter)
        
    
    best_p_removes_mean = np.mean(best_p_removes_iters)
    best_phi1_mean = np.mean(best_phi1_iters)
    best_phi2_mean = np.mean(best_phi2_iters)
    # print (best_phi1_mean)
    
    best_p_removes.append(best_p_removes_mean)
    best_phi1.append(best_phi1_mean)
    best_phi2.append(best_phi2_mean)
print (mp_distance_iters_p)
print (best_distance_iters)
'''
