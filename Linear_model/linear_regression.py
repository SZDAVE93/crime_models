# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:41:51 2018

@author: szdave
"""

import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as Init


class Linear_model(nn.Module):
    
    def __init__(self, dim_in):
        
        super(Linear_model, self).__init__()
        self.linear = nn.Linear(dim_in, 1, bias=False)
        Init.constant_(self.linear.weight.data, 1e-2)
        
    def forward(self, inputs):
        
        outputs = self.linear.forward(inputs)
        outputs = outputs[:, :, 0]
        return outputs


def train(ccrf_x, ccrf_y, epcho, lr, threshold):
    '''
    train a linear model
    '''
    t, m, n = ccrf_x.shape
    t_loss = np.inf
    
    m_model = Linear_model(n)
    m_loss = torch.nn.MSELoss()
    m_optimizer = torch.optim.SGD(m_model.parameters(), lr=lr)
    
    inputs = torch.from_numpy(ccrf_x).float()
    targets = torch.from_numpy(ccrf_y.T).float()
    
    step = 1
    for ite in range(0, epcho):
        #loss = 0
        m_optimizer.zero_grad()
        outputs = m_model.forward(inputs)
        loss = m_loss(outputs, targets)
        loss.backward()
        m_optimizer.step()
        if ite%step == 0:
            #print(loss.data[0])
            if np.abs(t_loss - loss.data[0])/step > threshold:
                t_loss = loss.data[0]
            else:
                print('minimized!')
                break

    return m_model

def predict(ccrf_x, m_model):
    '''
    predict
    '''
    inputs = torch.from_numpy(ccrf_x).float()
    outputs = m_model.forward(inputs)
    predicts = outputs.detach().numpy()
    return predicts