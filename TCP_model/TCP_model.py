# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:44:41 2018

@author: szdave
"""
import torch
import numpy as np
from TCP_model import ADMM_TCP

def build_P(kernerl_distance, N):
    
    P = np.zeros([N, N**2])
    for i in range(0, N):
        for j in range(0, N):
            if i == j:
                continue
            else:
                P[i, i*N+j] = kernerl_distance[i, j]
                P[j, i*N+j] = - kernerl_distance[i, j]
    return P

def build_Q(m_lambda, K):
    
    Q = np.zeros([K, K-1])
    for i in range(0, K-1):
        Q[i, i] = m_lambda
        Q[i+1, i] = - m_lambda
    return Q

def Predict(W, data_x, W_model, lag_days):
    
    K, N, M = data_x.shape
    Len_w = W.shape[0]
    new_W = np.zeros([M, N, K+lag_days])
    for i in range(0, lag_days):
        new_W[:, :, i] = W[Len_w-lag_days+i, :, :]
    predict_Y = np.zeros([N, K])
        
    for i in range(lag_days, K+lag_days):
        in_W = torch.from_numpy(new_W[:, :, i-lag_days:i]).float()
        in_X = torch.from_numpy(data_x[i-lag_days, :, :]).float()
        out_W = W_model.forward(in_W)
        out_Y = torch.diag(in_X.mm(out_W[:, :, 0])).reshape([N, 1])
        predict_Y[:, i-lag_days] = out_Y.detach().numpy()[:, 0]
        new_W[:, :, i] = out_W.detach().numpy()[:, :, 0]
    
    return predict_Y

def train(lr, epcho, train_X, train_Y, A_map_kernels, lag_days):
    
    K, N, M = train_X.shape
    m_lambda = 1e-1 # temporal relationship factor: lagging correlaion
    m_theta = 1e-3 # control the influence of regulation
    
    matrix = A_map_kernels[:,:,1]
    TCP_P = build_P(matrix, N)
    TCP_Q = build_Q(m_lambda, K)
    
    print("learning weights:")
    TCP_model = ADMM_TCP.TCP(train_Y, train_X, TCP_P, TCP_Q, theta=m_theta)
    TCP_W = TCP_model.ADMM_Optimization(lr=lr, epcho=epcho, threshold_wkn=1e-7)
    
    print("learning weights correlations:")
    m_model = TCP_model.W_Optimization(lag_days, TCP_W, epcho=epcho)
    
    #print("predicting:")
    #predict_Y = TPC_model.Predict(TPC_W, eval_X, m_model, lag_days)
    
    return TCP_W, m_model
    
    
