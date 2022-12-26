# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:23:12 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 16:36:54 2017

@author: Administrator
"""
import numpy as np

def rho_model(num_model,bins_real,centers_real,pdf_real):
    phi1_list = [0.99*b_i/num_model for b_i in range(1,num_model+1,1)]
    phi2_list = [0.99*b_i/num_model for b_i in range(1,num_model+1,1)]
    for i in range(len(pdf_real)):
        if (pdf_real[i]<=0):
            pdf_real[i]=0.0001
    distance_JS = np.ones([1,num_model*num_model],dtype=np.float64)*100#
    min_distance_JS = 100
    min_index = 0     
    pdf_model = np.zeros([num_model*num_model,bins_real],dtype=np.float64)
    num = 0
    opt_phi1 = 0
    opt_phi2 = 0
    
    for phi1 in phi1_list:
        for phi2 in phi2_list: 
            for bin_num in range(bins_real):
                z = centers_real[bin_num]
                F = [0,0,0,0]
                F[0] = 1.0*(phi1*phi2)*(z**2)
                F[1] = (phi1+phi2-2.0*phi1*phi2)*z
                F[2] = phi1*phi2-phi1-phi2-z+1
                F[3] = 1
                G_z = np.roots(F)
    #            G_z = 1.0*(M_z+1)/z
                pdf_model_bin = -1.0*np.imag(G_z)/np.pi
    #            pdf_model_bin_sel = pdf_model_bin[0]
    #            for pdf_model_bin_num in pdf_model_bin:
    #                if (np.abs(pdf_model_bin_sel-pdf_real[bin])>np.abs(pdf_model_bin_num-pdf_real[bin])):
    #                    pdf_model_bin_sel = pdf_model_bin_num
                pdf_model_bin_sel = np.max(pdf_model_bin)
                if (pdf_model_bin_sel<=0):#概率密度>0
                    pdf_model[num,bin_num] = 0.0001
                else:
                    pdf_model[num,bin_num] = pdf_model_bin_sel
            M = 0.5*(pdf_real+pdf_model[num])
            distance_JS[0,num] = 0.5*np.sum(pdf_real*np.log(1.0*pdf_real/M))+0.5*np.sum(pdf_model[num]*np.log(1.0*pdf_model[num]/M))#JS distance
            if (distance_JS[0,num]<min_distance_JS):
                min_distance_JS = distance_JS[0,num]
                opt_phi1 = phi1
                opt_phi2 = phi2
                min_index = num
            num = num+1
    min_distance_JS = distance_JS[0,min_index]
    opt_pdf_model = pdf_model[min_index]
    
    return min_distance_JS,opt_pdf_model,opt_phi1,opt_phi2