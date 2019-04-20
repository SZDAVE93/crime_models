# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:46:46 2018

@author: szdave
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as Init


class mLSTM(nn.Module):
    
    # inputs.shape: t * n, t = dim_batch, n = dim_inputs
    def __init__(self, dim_batch, dim_inputs, dim_hidden):
        
        super(mLSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(dim_inputs, dim_hidden)
        self.lstm2 = nn.LSTMCell(dim_hidden, dim_hidden)
        self.hidden_linear = nn.Linear(dim_hidden, dim_inputs, bias=True)
        self.time_linear = nn.Linear(dim_batch, 1)
        
        t_constant = 1e-2
        Init.constant_(self.lstm1.weight_hh.data, t_constant)
        Init.constant_(self.lstm1.weight_ih.data, t_constant)
        Init.constant_(self.lstm2.weight_hh.data, t_constant)
        Init.constant_(self.lstm2.weight_ih.data, t_constant)
        Init.constant_(self.hidden_linear.weight.data, t_constant)
        Init.constant_(self.time_linear.weight.data, t_constant)
        
        self.h_t1 = torch.zeros(dim_batch, dim_hidden)
        self.c_t1 = torch.zeros(dim_batch, dim_hidden)
        self.h_t2 = torch.zeros(dim_batch, dim_hidden)
        self.c_t2 = torch.zeros(dim_batch, dim_hidden)
        
        
    def forward(self, inputs):
        
        epcho, t, n = inputs.shape
        outputs = torch.zeros(epcho, n, t)
        for i in range(0, epcho):
            self.h_t1, self.c_t1 = self.lstm1(inputs[i,:,:], (self.h_t1, self.c_t1))
            self.h_t2, self.c_t2 = self.lstm2(self.h_t1, (self.h_t2, self.c_t2))
            outputs[i, :, :] = torch.t(self.hidden_linear.forward(self.h_t2))
        outputs = self.time_linear.forward(outputs)[:,:,0]
        return outputs
        

def train(ccrf_x, ccrf_y, dim_hidden, epcho, lstm_lr, threshold):
    
    t, n, m = ccrf_x.shape

    m_LSTM = mLSTM(m, n, dim_hidden) 
    m_loss = torch.nn.MSELoss()
    m_optimizer = torch.optim.SGD(m_LSTM.parameters(), lr=lstm_lr)
    t_loss = np.inf
    inputs = np.zeros([t, m, n])
    
    for i in range(t):
        inputs[i, :, :] = ccrf_x[i, :, :].T
    
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(ccrf_y.T).float()
    
    for i in range(0, epcho):
        m_optimizer.zero_grad()
        outputs = m_LSTM.forward(inputs)
        loss = m_loss(outputs, targets)
        loss.backward(retain_graph=True)
        m_optimizer.step()
        #print(loss.data[0])
        if t_loss > loss.data[0] and np.abs(t_loss - loss.data[0]) > threshold:
            t_loss = loss.data[0]
        else:
            print("Done!")
            break
    
    return m_LSTM

def predict(model, eval_x):
    
    t, n, m = eval_x.shape
    inputs = np.zeros([t, m, n])
    for i in range(t):
        inputs[i, :, :] = eval_x[i, :, :].T
    inputs = torch.from_numpy(inputs).float()
    
    predicts = model.forward(inputs).detach().numpy()
    return predicts

    
