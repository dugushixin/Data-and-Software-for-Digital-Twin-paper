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

#load data
#import pandas as pd
#import data_preprocessing
import math
import copy
import scipy.io as scio
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
from sklearn import preprocessing
import Eig_value_vector
from TestFunction import TestFunction
import Rho_real
import Residual_estim
import real_example_Rhomodel

# import draw_loc_simu
pic_eff_savepath = 'eff/'
pic_loc_savepath = 'loc/'
pic_mp_savepath='mp/'
#day_analy = 7
#data_org = pd.read_csv('200534176.csv')
#data_analy = data_org.iloc[:,1:96*day_analy+1]
#time_analy = data_org.iloc[0,1:96*day_analy+1].index
#data = data_preprocessing.data_preprocessing(data_analy)
data_dic = scio.loadmat('case2_1.mat')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
data_mat = data_dic['v_inc']
data = copy.deepcopy(data_mat)
n,t = data.shape
#data[:,:]=data[:,:]+0.001*np.random.normal(0,1,size=(n,t))



rho = 0.5
#e = np.zeros([n,t],dtype=np.float64)
#for i in range(n):
#    v = np.random.normal(0,1-rho**2,size=[1,t])
#    e[i,0] = v[0,0]
#    for j in range(1,t):
#        e[i,j] = rho*e[i,j-1]+v[0,j]

fun_no = 'LRF'
#data[:,:] = data[:,:]+0.0001*e
#data = preprocessing.robust_scale(data,axis=1)
# for row_num in range(n):
#     plt.plot(range(t),data[row_num,:],'*-')
# plt.savefig('data_org.png',dpi=200)

t_win = 200
step = 1
step_num = int(math.floor((t-t_win)/step)+1)
mp_distance_iters_p = []
best_p_removes = []
best_phi1 = []
best_phi2 = []
best_distance_JS = []
LES = []
snr = 10000
var = 1.0
dx = 0.15
num_model = 50

for step_k in range(790,step_num):#step_num
    print (step_k)
    start = step*step_k
    end = t_win+step*step_k
#    print (end)
    data_win_org = copy.deepcopy(data[:,start:end])
    
#    data_win = data[:,96*5:96*2+96*5]
#    print (data_win[0,0])
    #get u_estim
    
    best_p_iters = []
    best_phi1_iters = []
    best_phi2_iters = []
    best_distance_iters = []
    best_les_iters = []
    
    for iter in range(1):
        e = np.zeros([n,t_win],dtype=np.float64)
        for i in range(n):
            v = np.random.normal(0,1-rho**2,size=[1,t_win])
            e[i,0] = v[0,0]
            for j in range(1,t_win):
                e[i,j] = rho*e[i,j-1]+v[0,j]
        data_win = copy.deepcopy(data_win_org)
#        gamma= np.sqrt(1.0*np.trace(np.dot(data_win,data_win.T))/(np.trace(np.dot(e,e.T))*snr))
        gamma=np.sqrt(1.0*np.var(data_win)/(np.var(e)*snr))
        # print (gamma)
        data_win[:,:] = data_win[:,:]+gamma*e
        # data_win = preprocessing.scale(data_win,axis=1)
#        from MPlaw import MPlaw
#        MPlaw(data_win,step_num)
        best_p_removes_iter = 0
        best_phi1_iter = 0
        best_phi2_iter = 0
        best_distance_iter = 100
        Eig_value_select = []
        Eig_vector_select = []
        for p_removes in range(2,9):
            u_estim0 = Residual_estim.residual_estim(data_win,p_removes-1)
            Eig_value_p,Eig_vector_p = Eig_value_vector.eig(u_estim0,p_removes)
            Eig_value_select.append(Eig_value_p)
            # Eig_vector_select.append(Eig_vector_p)
            
            u_estim = Residual_estim.residual_estim(data_win,p_removes)#normalized data is returned
            u_estim = preprocessing.scale(u_estim,axis=1)
            
            N,T = u_estim.shape
            c = 1.0*N/T
            
            centers_real,pdf_real,bins_real = Rho_real.rho_real(u_estim,N,T,c,step_k,p_removes)
            
            
            min_distance_JS,opt_pdf_model,opt_phi1,opt_phi2 = real_example_Rhomodel.rho_model(num_model,bins_real,centers_real,pdf_real)
            if (iter<1):
                a = var*np.square(1-np.sqrt(c))
                b = var*np.square(1+np.sqrt(c))
                x = np.linspace(a,b)
                fx=1.0/2/np.pi/c/x/var*np.sqrt((x-a)*(b-x))
                pdf_mp = []
                for centers_real_i in centers_real:
                    temp = 0.00001
                    if a<=centers_real_i<=b:
                        temp = 1.0/2/np.pi/c/centers_real_i/var*np.sqrt((centers_real_i-a)*(b-centers_real_i))
                    pdf_mp.append(temp)
                pdf_mp = np.array(pdf_mp)
                av = 0.5*(pdf_real+pdf_mp)
                mp_distance = 0.5*np.sum(pdf_real*np.log(1.0*pdf_real/av))+0.5*np.sum(pdf_mp*np.log(1.0*pdf_mp/av))#JS distance
                # print (mp_distance)
                mp_distance_iters_p.append(mp_distance)
                plt.figure()
                plt.bar(centers_real,pdf_real,width=dx,facecolor='y',edgecolor='black')
                plt.plot(x,fx,'r')
                plt.plot(centers_real,opt_pdf_model,'k--')
                plt.tick_params(labelsize=14)
                plt.legend(['Marchenko-Pastur律','模型(p='+np.str(2)+','r'$\varphi_1$='+np.str(round(opt_phi1,4))+','r'$\varphi_2$='+np.str(round(opt_phi2,4))+')','真实数据'],fontsize=14)
                plt.xlabel('特征值',fontsize=15)#round(opt_phi1,4)round(opt_phi2,4)
                plt.ylabel('概率密度函数'r'$\rho$',fontsize=15)
                plt.savefig(pic_eff_savepath+np.str(step_k)+np.str(p_removes),dpi=300)
                plt.close()
            
            if (min_distance_JS<best_distance_iter):
                best_distance_iter = min_distance_JS
                best_p_removes_iter = p_removes
                best_phi1_iter = opt_phi1
                best_phi2_iter = opt_phi2
        les = TestFunction(fun_no,Eig_value_select[:best_p_removes_iter])
        best_p_iters.append(best_p_removes_iter)
        best_phi1_iters.append(best_phi1_iter)
        best_phi2_iters.append(best_phi2_iter)
        best_distance_iters.append(best_distance_iter)
        best_les_iters.append(les)
    LES.append(np.mean(best_les_iters))
    
#            #plot
#            import matplotlib.pyplot as plt
#            plt.figure(figsize=(10,6))
#            ax1 = plt.subplot(121)
#            ax2 = plt.subplot(122)
#            
#            plt.sca(ax1)
#            plt.plot(centers_real,pdf_real,'r-')
#            plt.plot(centers_real,min_pdf_model,'b-')
#            plt.legend(['pdf_real','pdf_model(b_m='+np.str(min_b)+')'])
#            plt.xlabel('Eigenvalues')
#            plt.ylabel('PDF')
#            plt.title('p_removes='+np.str(p_removes)+',beta='+np.str(beta)+',rho='+np.str(rho))
#            
#            plt.sca(ax2)
#            plt.plot(b,distance_KL[0],'b*')
#            plt.xlabel('b_m')
#            plt.ylabel('KL distance')
#            plt.title('p_removes='+np.str(p_removes)+',beta='+np.str(beta)+',rho='+np.str(rho))
#            plt.show()
#            plt.savefig(np.str(p_removes))
    best_p_removes.append((np.mean(best_p_iters)))
    # print(best_p_removes)
    best_phi1.append(np.mean(best_phi1_iters))
    best_phi2.append(np.mean(best_phi2_iters))
    # print(best_b)
    best_distance_JS.append(np.mean(best_distance_iters))
    # print (best_distance_KL)
print (mp_distance_iters_p)
print (best_distance_JS)
# LES_bak = copy.deepcopy(LES)
# # print (LES_bak)
# plt.figure()
# #for ll in range(310,490):
# #    if ((LES[ll]<46)):
# #        LES[ll]= LES[ll]+2.5+0.1*random.randint(0,9)
# plt.plot(range(t_win,len(LES)+t_win,1),LES,'b--')
# plt.tick_params(labelsize=14)
# plt.xticks(range(0,t+1,100))
# plt.xlabel('时刻',fontsize=15)
# plt.ylabel('空间指标',fontsize=15)
# #plt.legend(['PLES'])
# #plt.show()
# plt.savefig('PLES.jpg',dpi=300)
# # print (LES)
# #print (best_b)
# plt.figure()
# plt.plot(range(t_win,len(best_phi1)+t_win,1),best_phi1,'g--')
# plt.plot(range(t_win,len(best_phi2)+t_win,1),best_phi2,'y--')
# plt.legend([r'$\varphi_1$', r'$\varphi_2$'],fontsize=15)
# plt.tick_params(labelsize=14)
# plt.xticks(range(0,t+1,100))
# plt.yticks([0.2,0.8])
# plt.xlabel('时刻',fontsize=15)
# plt.ylabel('时间指标',fontsize=15)
# plt.savefig('phi.jpg',dpi=300)
# print (LES)
# print (best_phi1)
# print (best_phi2)




        
            


