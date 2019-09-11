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
        if len(outputs.shape) == 3:
            outputs = outputs[:, :, 0]
        return outputs
    
    def predict_global(self, eval_x, m_model):
        '''
        predict
        '''
        inputs = torch.from_numpy(eval_x).float()
        outputs = m_model.forward(inputs)
        predicts = outputs.detach().numpy().T
        return predicts
    
    def predict_iter(self, eval_x, m_models):
        
        t, num_regions, seq_len = eval_x.shape
        inputs = torch.from_numpy(eval_x[0, :, :].reshape(num_regions, seq_len)).float()
        pred_y = m_models.forward(inputs).detach().numpy()
        iter_x = np.append(eval_x[0, :, :].reshape(num_regions, seq_len)[:, 1:seq_len], pred_y, axis=1)
        for i in range(1, t):
            inputs = torch.from_numpy(iter_x).float()
            pred_y_tmp = m_models.forward(inputs).detach().numpy()
            pred_y = np.append(pred_y, pred_y_tmp, axis=1)
            iter_x = np.append(iter_x[:, 1:seq_len], pred_y_tmp, axis=1)
        return pred_y

def train(train_x, train_y, lr, iters, threshold):
    '''
    train a linear model
    '''
    t, m, n = train_x.shape
    t_loss = np.inf
    
    models = []
    for i in range(0, m):
        m_model = Linear_model(n)
        m_loss = torch.nn.MSELoss()
        m_optimizer = torch.optim.SGD(m_model.parameters(), lr=lr)
        
        inputs = torch.from_numpy(train_x[:, i, :]).float()
        targets = torch.from_numpy(train_y.T[:, i].reshape(t, 1)).float()
        
        for ite in range(0, iters):
            m_optimizer.zero_grad()
            outputs = m_model.forward(inputs)
            loss = m_loss(outputs, targets)
            loss.backward()
            m_optimizer.step()
            print(loss.data[0])
            if np.abs(t_loss - loss.data[0]) > threshold:
                t_loss = loss.data[0]
            else:
                print('minimized!')
                break
        models.extend([m_model])

    return models

def train_global(train_x, train_y, lr, iters, threshold):
    '''
    train a linear model
    '''
    t, m, n = train_x.shape
    t_loss = np.inf
    
    m_model = Linear_model(n)
    m_loss = torch.nn.MSELoss()
    m_optimizer = torch.optim.SGD(m_model.parameters(), lr=lr)
    
    inputs = torch.from_numpy(train_x).float()
    targets = torch.from_numpy(train_y.T).float()
    
    step = 1
    for ite in range(0, iters):
        #loss = 0
        m_optimizer.zero_grad()
        outputs = m_model.forward(inputs)
        loss = m_loss(outputs, targets)
        loss.backward()
        m_optimizer.step()
        if ite%step == 0:
            print(loss.data[0])
            if np.abs(t_loss - loss.data[0])/step > threshold:
                t_loss = loss.data[0]
            else:
                print('minimized!')
                break

    return m_model

def predict(eval_x, m_model):
    '''
    predict
    '''
    inputs = torch.from_numpy(eval_x).float()
    outputs = m_model.forward(inputs)
    predicts = outputs.detach().numpy()
    return predicts