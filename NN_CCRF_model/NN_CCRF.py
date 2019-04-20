# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:14:59 2018

@author: szdave
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as Init

class NN_CCRF(nn.Module):
    
    def __init__(self, feature_model, dim_inputs, num_layer):
        
        super(NN_CCRF, self).__init__()
        self.feature_model = feature_model
        self.kernel = nn.Linear(dim_inputs, dim_inputs, bias=True)
        self.weights_feature_pairwise = nn.Linear(2, 1, bias=True)
        
        m_constant = 1e-2
        Init.constant_(self.kernel.weight.data, m_constant)
        Init.constant_(self.weights_feature_pairwise.weight.data, m_constant)
        # building the iteration process as an RNN framework
        self.pairwise_model = nn.Sequential()
        for i in range(num_layer):
            self.pairwise_model.add_module("K"+str(i), self.kernel)
        
    def forward(self, inputs):
        
        # inputs should have the required dimension from different feature model
        # feature model should always output data in t * n dimensional tensor
        item_feature = self.feature_model.forward(inputs)
        item_pairwise = self.pairwise_model.forward(item_feature)
        
        item_feature = torch.unsqueeze(item_feature, 2)
        item_pairwise = torch.unsqueeze(item_pairwise, 2)
        items = torch.cat((item_feature, item_pairwise), 2)
        outputs = self.weights_feature_pairwise.forward(items)
        outputs = outputs[:,:,0]
        
        #outputs = item_pairwise
        return outputs
    
def train_m(feature_model, ccrf_x, ccrf_y, rnn_layers, epcho, lr, threshold):
    
    t, n, m = ccrf_x.shape
    if n > m:
        dim_in = n
    else:
        dim_in = m
    m_model = NN_CCRF(feature_model, dim_in, rnn_layers)
    m_loss = torch.nn.MSELoss()
    m_optimizer = torch.optim.SGD(m_model.parameters(), lr=lr)
    t_loss = np.inf
    
    inputs = torch.from_numpy(ccrf_x).float()
    targets = torch.from_numpy(ccrf_y.T).float()
    
    
    for i in range(epcho):
        m_optimizer.zero_grad()
        outputs = m_model.forward(inputs)
        loss = m_loss(outputs, targets)
        loss.backward(retain_graph=True)
        m_optimizer.step()
        #print(loss.data[0])
        if t_loss > loss.data[0] and np.abs(t_loss - loss.data[0]) > threshold:
            t_loss = loss.data[0]
        else:
            print("Done!")
            break
    
    return m_model

def predict_m(model, eval_x):
    
    inputs = torch.from_numpy(eval_x).float()
    outputs = model.forward(inputs)
    predicts = outputs.detach().numpy()
    return predicts
    
        
                
        
    
        
            