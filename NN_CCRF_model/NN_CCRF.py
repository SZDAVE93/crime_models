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
        self.feature_model = feature_model.cuda()
        self.kernel = nn.Linear(dim_inputs, dim_inputs, bias=True).cuda()
        self.weights_feature_pairwise = nn.Linear(2, 1, bias=True).cuda()
        
        m_constant = 1e-2
        Init.constant_(self.kernel.weight.data, m_constant)
        Init.constant_(self.weights_feature_pairwise.weight.data, m_constant)
        # building the iteration process as an RNN framework
        self.pairwise_model = nn.Sequential().cuda()
        for i in range(num_layer):
            self.pairwise_model.add_module("K"+str(i), self.kernel)
        
    def forward(self, inputs):
        
        # inputs should have the required dimension from different feature model
        # feature model should always output data in t * n dimensional tensor
        item_feature = self.feature_model.forward(inputs)
        item_pairwise = self.pairwise_model.forward(item_feature)
        
        item_feature = torch.unsqueeze(item_feature, 2)
        item_pairwise = torch.unsqueeze(item_pairwise, 2)
        items = torch.cat((item_feature, item_pairwise), 2).cuda()
        outputs = self.weights_feature_pairwise.forward(items)
        outputs = outputs[:,:,0]
        
        #outputs = item_pairwise
        return outputs
    
    def predict(self, eval_x, m_model):
    
        inputs = torch.from_numpy(eval_x).float().cuda()
        outputs = m_model.forward(inputs).cpu()
        predicts = outputs.detach().numpy().T
        return predicts
    
    def predict_iter(self, eval_x, m_model):
        
        t, seq_len, num_regions = eval_x.shape
        eval_x = eval_x.transpose(1, 0, 2)
        inputs = torch.from_numpy(eval_x[:, 0, :].reshape(seq_len, 1, num_regions)).float().cuda()
        pred_y = m_model.forward(inputs).cpu().detach().numpy()
        iter_x = np.append(eval_x[:, 0, :].reshape(seq_len, num_regions)[1:seq_len], pred_y, axis=0)
        
        for i in range(1, t):
            inputs = torch.from_numpy(iter_x.reshape(seq_len, 1, num_regions)).float().cuda()
            pred_y_tmp = m_model.forward(inputs).cpu()
            pred_y_tmp = pred_y_tmp.detach().numpy()
            pred_y = np.append(pred_y, pred_y_tmp, axis=0)
            iter_x = np.append(iter_x[1:seq_len], pred_y_tmp, axis=0)
        pred_y = pred_y.T
        return pred_y
    
def train_m(train_x, train_y, lr, iters, threshold, feature_model, rnn_layers):
    
    dim_in = train_y.shape[0]
    m_model = NN_CCRF(feature_model, dim_in, rnn_layers)
    m_loss = torch.nn.MSELoss()
    m_optimizer = torch.optim.SGD(m_model.parameters(), lr=lr)
    t_loss = np.inf
    
    inputs = torch.from_numpy(train_x).float().cuda()
    targets = torch.from_numpy(train_y.T).float().cuda()
    
    
    for i in range(iters):
        m_optimizer.zero_grad()
        outputs = m_model.forward(inputs)
        loss = m_loss(outputs, targets)
        loss.backward(retain_graph=True)
        m_optimizer.step()
        print(loss.data[0])
        if t_loss > loss.data[0] and np.abs(t_loss - loss.data[0]) > threshold:
            t_loss = loss.data[0]
        else:
            print("minimized!")
            break
        
    
    return m_model

def predict_m(eval_x, m_model):
    
    inputs = torch.from_numpy(eval_x).float().cuda()
    outputs = model.forward(inputs).cpu()
    predicts = outputs.detach().numpy()
    return predicts
    
        
                
        
    
        
            