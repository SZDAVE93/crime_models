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

def predict_iter(eval_x, m_model, lag_days):
    
    W = m_model[0]
    W_model = m_model[1]
    K, N, M = eval_x.shape
    Len_w = W.shape[0]
    new_W = np.zeros([M, N, K+lag_days])
    for i in range(0, lag_days):
        new_W[:, :, i] = W[Len_w-lag_days+i, :, :]
    pred_y = np.zeros([N, 1])
    iter_x = eval_x[0, :, :]
        
    for i in range(lag_days, K+lag_days):
        in_W = torch.from_numpy(new_W[:, :, i-lag_days:i]).float()
        in_X = torch.from_numpy(iter_x).float()
        out_W = W_model.forward(in_W)
        out_Y = torch.diag(in_X.mm(out_W[:, :, 0])).reshape([N, 1])
        pred_y_tmp = out_Y.detach().numpy()
        pred_y = np.append(pred_y, pred_y_tmp, axis=1)
        new_W[:, :, i] = out_W.detach().numpy()[:, :, 0]
        iter_x = np.append(iter_x[:, 1:M], pred_y_tmp, axis=1)
    
    pred_y = pred_y[:, 1:K+1]
    return pred_y

def train(train_x, train_y, lr, iters, threshold, tcp_kernel, lag_days, lamda, theta):
    
    K, N, M = train_x.shape
    m_lambda = 10**(-lamda) # temporal relationship factor: lagging correlaion
    m_theta = 10**(-theta) # control the influence of regulation
    
    matrix = tcp_kernel
    TCP_P = build_P(matrix, N)
    TCP_Q = build_Q(m_lambda, K)
    
    print("learning weights:")
    TCP_model = ADMM_TCP.TCP(train_y, train_x, TCP_P, TCP_Q, theta=m_theta)
    TCP_W = TCP_model.ADMM_Optimization(lr=lr, epcho=iters, threshold_wkn=threshold)
    
    print("learning weights correlations:")
    m_model = TCP_model.W_Optimization(lag_days, TCP_W, lr=lr, epcho=iters)
    
    #print("predicting:")
    #predict_Y = TPC_model.Predict(TPC_W, eval_X, m_model, lag_days)
    
    return (TCP_W, m_model)
    
    
