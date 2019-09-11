# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:42:17 2018

@author: szdave
"""

import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.init as Init
import torch.nn.functional as F

class CRF_RNN(nn.Module):
    
    # gird based chichago 5*7 = 35 New York 7*9 = 63
    def __init__(self, rnn_num, F_in, D_in, latent=False, kernel_size=(63, 63)):
        '''
        D_in: the number of kernels
        '''
        super(CRF_RNN, self).__init__()
        self.latent = latent
        dim_factor = kernel_size[0]
        self.latent_factor = nn.ModuleList([nn.Linear(dim_factor, dim_factor, bias=False).cuda() for j in range(D_in)])
        self.feature_linear_layer = nn.Linear(F_in[0], F_in[1], bias=False).cuda()
        self.linear_layer = nn.Linear(D_in, 1, bias=False).cuda()
        self.linear_factor = nn.Linear(1, 1, bias=False).cuda()
        # parameter initialize
        constant = True
        self.cons_a = 1e-2
        self.cons_b = 1e-2
        if constant:
            self.contant_init(D_in, dim_factor, self.cons_a, self.cons_b)
        else:
            self.norml_init(D_in, 1e-2, 1e-2)
        self.Con1D = F.conv1d
        
        self.rnn_num = rnn_num
    
    def contant_init(self, D_in, dim_n, constant_a, constant_b):

        for i in range(D_in):
            Init.constant_(self.latent_factor[i].weight.data, constant_a)
        Init.constant_(self.feature_linear_layer.weight.data, constant_a)
        Init.constant_(self.linear_layer.weight.data, constant_b)
        #Init.constant_(self.linear_factor.weight.data, constant_b)
    
    def norml_init(self, D_in, mean_a, mean_b):
        
        std = 0.01
        for i in range(D_in):
            Init.normal_(self.latent_factor[i].weight.data, mean_a, std)
        Init.normal_(self.feature_linear_layer.weight.data, mean_a, std)
        Init.normal_(self.linear_layer.weight.data, mean_b, std)
        Init.normal_(self.linear_factor.weight.data, mean_b, std)
        
    
    def kernel_modification_layer(self, g_kernels):
        '''
        factors for each kernel that modify the weights
        '''
        n, n, k = g_kernels.shape
        kernels = torch.randn(n, n, k).cuda()
        for i in range(0, k):
            kernels[:, :, i] = self.latent_factor[i].forward(g_kernels[:,:,i])
        return kernels
    
    def static_conv_layer(self, inputs, g_kernels):
        '''
        calculate the results of inputs convoluted by M static filters (Gaussian kernels)
        
        Shape:
            inputs: T * N * 1
            g_kernels: N * N * M
            Output:
                convoluted results, the shape is T * N * M
        '''
        n = g_kernels.shape[0]
        k = g_kernels.shape[2]
        t = inputs.shape[0]
        outputs = torch.zeros(t, n, k).cuda()
        for i in range(0, k):
            t_kernel = g_kernels[:, :, i].view(n, n, 1).cuda()
            t_result = self.Con1D(inputs, t_kernel)
            outputs[:, :, i] = t_result[:, :, 0]
    
        return outputs
    
    def feature_layer(self, inputs):
        
        outputs = self.feature_linear_layer.forward(inputs)
        return outputs
    
    def linear_combine(self, inputs):
        
        t, n, k = inputs.shape
        outputs = torch.zeros(t, n, 1).cuda()
        for i in range(0, n):
            outputs[:, i, :] = self.linear_layer[i].forward(inputs[:, i, :])
        
        return outputs
            
    def forward(self, inputs, kernels):
        
        self.crf_kernels = kernels
        t, n, m = inputs.shape
        inputs = self.feature_layer(inputs)
        all_ones = torch.ones(t, n, 1).cuda()
        feature_ones = torch.ones(t, n, m).cuda()
        f_weights = self.feature_layer(feature_ones)
        orginals = inputs
            
        for k in range(0, self.rnn_num):
            if self.latent:
                kernels = self.kernel_modification_layer(kernels)
            layer_11 = self.static_conv_layer(inputs, kernels)
            layer_12 = self.static_conv_layer(all_ones, kernels)
            layer_21 = 2 * self.linear_layer.forward(layer_11)
            layer_22 = 2 * self.linear_layer.forward(layer_12)
            layer_31 = orginals + layer_21
            layer_32 = f_weights + layer_22
            inputs = layer_31 / layer_32
            
            outputs = inputs
        
        return outputs
    
    def predict(self, eval_x, m_model):
        
        linear_predicts = torch.from_numpy(eval_x).float().cuda()
        crf_kernels = m_model.crf_kernels.cuda()
        
        results = m_model.forward(linear_predicts, crf_kernels).cpu()
        results = results.detach().numpy()[:,:,0].T
    
        return results
    
    def predict_iter(self, eval_x, m_model):
        
        t, num_regions, seq_len = eval_x.shape
        iter_x = eval_x[0, :, :].reshape([1, num_regions, seq_len])
        
        pred_y = np.zeros([num_regions, 1])
        crf_kernels = m_model.crf_kernels.cuda()
        for i in range(0, t):
            linear_predicts = torch.from_numpy(iter_x).float().cuda()
            pred_y_tmp = m_model.forward(linear_predicts, crf_kernels).cpu()
            pred_y_tmp = pred_y_tmp.detach().numpy()[:,:,0].T
            pred_y = np.append(pred_y, pred_y_tmp, axis=1)
            iter_x = np.append(iter_x[:, :, 1:seq_len], pred_y_tmp.reshape([1, num_regions, 1]), axis=2)
    
        pred_y = pred_y[:, 1:t+1]
        return pred_y
        

def train(train_y, train_x, lr, iters, threshold, crf_kernels, rnn_layers):
    
    t, n, m = train_x.shape
    M = crf_kernels.shape[2]
    F = (m, 1)
    
    m_crfRnn = CRF_RNN(rnn_layers, F, M, latent=True, 
                       kernel_size=(crf_kernels.shape[0], crf_kernels.shape[0]))
    
    
    m_loss = torch.nn.MSELoss()
    m_optimizer = torch.optim.SGD(m_crfRnn.parameters(), lr=lr)
    t_loss = np.inf
    
    inputs = torch.from_numpy(train_x).float().cuda()
    targets = torch.from_numpy(train_y.T).float().reshape([t, n, 1]).cuda()
    crf_kernels = torch.from_numpy(crf_kernels).float().cuda()
    
    for i in range(0, iters):
        m_optimizer.zero_grad()
        outputs = m_crfRnn.forward(inputs, crf_kernels)
        loss = m_loss(outputs, targets)
        loss.backward()
        m_optimizer.step()
        print(loss.data[0])
        if np.abs(t_loss - loss.data[0]) > threshold and t_loss > loss.data[0]:
            t_loss = loss.data[0]
        else:
            print('minimized!')
            break
    
    return m_crfRnn

def predict(eval_x, model, crf_kernels):
    
    linear_predicts = torch.from_numpy(eval_x).float().cuda()
    crf_kernels = torch.from_numpy(crf_kernels).float().cuda()
        
    results = model.forward(linear_predicts, crf_kernels).cpu()
    results = results.detach().numpy()[:,:,0].T
    
    return results

def ranking_MAP(data, result, rank_k):
    
    n, t = data.shape
    m_MAP = np.zeros([rank_k, 1])
    for i in range(0, t):
        sort_data = sorted(enumerate(data[:, i]), key = lambda x:x[1], reverse=True)
        sort_result = sorted(enumerate(result[:, i]), key = lambda x:x[1], reverse=True)
        sort_data = np.array(sort_data)
        sort_result = np.array(sort_result)
        flags = sort_data[:, 0] == sort_result[:, 0]
        flags = flags.reshape([n, 1])
        for j in range(0, rank_k):
            temp = flags[0:j+1].reshape([1, j+1])
            m_MAP[j] = m_MAP[j] + len(np.where(temp == True)[0]) / (j+1)
    m_MAP = m_MAP / t
    return m_MAP
        