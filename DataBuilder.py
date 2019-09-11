# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:41:03 2019

@author: yifei
"""

import numpy as np
import scipy.stats as ss


class d_Builder():
    
    def __init__(self, data_path, ar_days, 
                 train_end, train_days, eval_days):
        self.path = data_path
        self.ar_days = ar_days
        self.train_end = train_end
        self.train_days = train_days
        self.eval_days = eval_days
        
    def load_xy(self, is_nnccrf):
        
        X = np.load('{}CCRF_X_{}.npy'.format(self.path, self.ar_days))
        Y = np.load('{}CCRF_Y_{}.npy'.format(self.path, self.ar_days))
        Train_X = X[self.train_end-self.train_days:self.train_end]
        Train_Y = Y[:, self.train_end-self.train_days:self.train_end]
        Eval_X = X[self.train_end:self.train_end+self.eval_days]
        Eval_Y = Y[:, self.train_end-1:self.train_end+self.eval_days]
        
        if is_nnccrf:
            Train_X = Train_X.transpose(0, 2, 1)
            Eval_X = Eval_X.transpose(0, 2, 1)
        
        return (Train_X, Train_Y), (Eval_X, Eval_Y)
    
    def load_kernel(self, kernel_name, simi_len):
        
        if kernel_name == 'dis':
            Y = np.load('{}CCRF_Y_{}.npy'.format(self.path, self.ar_days))
            kernel = self.dis_kernel(Y[:, self.train_end-simi_len:self.train_end])
        else:
            kernel = np.load('{}kernel_{}.npy'.format(self.path, kernel_name))
        kernel = kernel[:, :, np.newaxis]
        return kernel
    
    def load_kernels(self, kernel_names, simi_len):
        
        kernels = self.load_kernel(kernel_names[0], simi_len)
        for i in range(1, len(kernel_names)):
            tmp_kernel = self.load_kernel(kernel_names[i], simi_len)
            kernels = np.append(kernels, tmp_kernel, axis=2)
        return kernels
    
    def load_tcp_kernel(self, kernel_name):
        
        if kernel_name == 'dis':
            kernel = np.load('{}kernel_tcp.npy'.format(self.path))
        else:
            kernel = self.load_kernel(kernel_name[0], -1)
            kernel = kernel[:, :, 0]
        return kernel

    def KL_distance(self, Yi, Yj, v_max):
            
        # re-orgnize the raw data
        m_bins = [x for x in range(0, v_max, 2)]
        # to prevent 0 items, manully add 0.1
        freq_p = np.histogram(Yi, bins=m_bins)[0] + 0.1
        freq_q = np.histogram(Yj, bins=m_bins)[0] + 0.1
        freq_p = freq_p / np.sum(freq_p)
        freq_q = freq_q / np.sum(freq_q)
        
        simi = np.exp(-np.mean([freq_p * np.log(freq_p / freq_q), freq_q * np.log(freq_q / freq_p)]))
        #simi = np.exp(-np.mean([ss.entropy(freq_p, freq_q), ss.entropy(freq_p, freq_q)]))
        return simi
    
    def dis_kernel(self, Y):
        
        n = Y.shape[0]
        kernel = np.zeros([n, n])
        
        for i in range(0, n):
            for j in range(i+1, n):
                tmp_max = np.max([np.max(Y[i, :]), np.max(Y[j, :])]).astype(int)
                simi = self.KL_distance(Y[i, :], Y[j, :], tmp_max)
                kernel[i, j] = simi
                kernel[j, i] = kernel[i, j]
        return kernel
            
            
            