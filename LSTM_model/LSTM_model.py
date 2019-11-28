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
    
    def __init__(self, dim_inputs, dim_hidden, batch_size, num_layers=2, 
                dim_output=77, seq_len=7):
        
        super(mLSTM, self).__init__()
        self.dim_inputs = dim_inputs
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dim_output = dim_output
        self.seq_len = seq_len
        
        self.lstm = nn.LSTM(self.dim_inputs, self.dim_hidden, self.num_layers).cuda()
        self.hidden_linear = nn.Linear(self.dim_hidden, self.dim_output).cuda()
        
    def forward(self, inputs):
        
        lstm_out, self.hidden_states = self.lstm(inputs)
        outputs = self.hidden_linear(lstm_out[-1])
        
        return outputs
    
    def predict_iter(self, eval_x, m_model):
        
        t, num_regions, seq_len = eval_x.shape
        eval_x = eval_x.transpose(2, 0, 1)
        inputs = torch.from_numpy(eval_x[:, 0, :].reshape(seq_len, 1, num_regions)).float().cuda()
        pred_y = m_model.forward(inputs).cpu().detach().numpy()
        iter_x = np.append(eval_x[:, 0, :].reshape(seq_len, num_regions)[1:seq_len], pred_y, axis=0)
        for i in range(1, t):
            inputs = torch.from_numpy(iter_x.reshape(seq_len, 1, num_regions)).float().cuda()
            pred_y_tmp = m_model.forward(inputs).cpu().detach().numpy()
            pred_y = np.append(pred_y, pred_y_tmp, axis=0)
            iter_x = np.append(iter_x[1:seq_len], pred_y_tmp, axis=0)
        pred_y = pred_y.T
        return pred_y

def train(train_x, train_y, lr, iters, threshold, dim_hidden, num_layers):
    
    batch_size, dim_input, seq_len = train_x.shape

    m_LSTM = mLSTM(dim_inputs=dim_input, dim_hidden=dim_hidden, batch_size=batch_size,
                   num_layers=num_layers, dim_output=dim_input)
    m_loss = torch.nn.MSELoss()
    m_optimizer = torch.optim.SGD(m_LSTM.parameters(), lr=lr)
    t_loss = np.inf
    inputs = train_x.transpose(2, 0, 1)
    
    inputs = torch.from_numpy(inputs).float().cuda()
    targets = torch.from_numpy(train_y.T).float().cuda()
    
    for i in range(0, iters):
        m_optimizer.zero_grad()
        outputs = m_LSTM.forward(inputs)
        loss = m_loss(outputs, targets)
        loss.backward(retain_graph=True)
        m_optimizer.step()
        if i%1 == 0:
            print(loss.data[0])
        if t_loss > loss.data[0] and np.abs(t_loss - loss.data[0]) > threshold:
            t_loss = loss.data[0]
        else:
            print("minimized!")
            break
    
    return m_LSTM    
