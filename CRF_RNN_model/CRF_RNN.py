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
        self.latent_factor = nn.ModuleList([nn.Linear(dim_factor, dim_factor, bias=False) for j in range(D_in)])
        self.feature_linear_layer = nn.Linear(F_in[0], F_in[1], bias=False)
        self.linear_layer = nn.Linear(D_in, 1, bias=False)
        self.linear_factor = nn.Linear(1, 1, bias=False)
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
        
    
    def kernel_modification_layer(self, g_kernels, cuda):
        '''
        factors for each kernel that modify the weights
        '''
        n, n, k = g_kernels.shape
        if cuda:
            kernels = torch.randn(n, n, k).cuda()
        else:
            kernels = torch.randn(n, n, k)
        for i in range(0, k):
            kernels[:, :, i] = self.latent_factor[i].forward(g_kernels[:,:,i])
        return kernels
    
    def static_conv_layer(self, inputs, g_kernels, cuda):
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
        if cuda:
            outputs = torch.zeros(t, n, k).cuda()
        else:
            outputs = torch.zeros(t, n, k)
        for i in range(0, k):
            t_kernel = g_kernels[:, :, i].view(n, n, 1)
            t_result = self.Con1D(inputs, t_kernel)
            outputs[:, :, i] = t_result[:, :, 0]
    
        return outputs
    
    def feature_layer(self, inputs):
        
        outputs = self.feature_linear_layer.forward(inputs)
        return outputs
    
    def linear_combine(self, inputs, cuda):
        
        t, n, k = inputs.shape
        if cuda:
            outputs = torch.zeros(t, n, 1).cuda()
        else:
            outputs = torch.zeros(t, n, 1)
        for i in range(0, n):
            outputs[:, i, :] = self.linear_layer[i].forward(inputs[:, i, :])
        
        return outputs
            
    def forward(self, inputs, kernels, cuda):
        
        t, n, m = inputs.shape
        inputs = self.feature_layer(inputs)
        if cuda:
            all_ones = torch.ones(t, n, 1).cuda()
            feature_ones = torch.ones(t, n, m).cuda()
        else:
            all_ones = torch.ones(t, n, 1)
            feature_ones = torch.ones(t, n, m)
        f_weights = self.feature_layer(feature_ones)
        orginals = inputs
            
        for k in range(0, self.rnn_num):
            if self.latent:
                kernels = self.kernel_modification_layer(kernels, cuda)
            layer_11 = self.static_conv_layer(inputs, kernels, cuda)
            layer_12 = self.static_conv_layer(all_ones, kernels, cuda)
            layer_21 = 2 * self.linear_layer.forward(layer_11)
            layer_22 = 2 * self.linear_layer.forward(layer_12)
            layer_31 = orginals + layer_21
            layer_32 = f_weights + layer_22
            inputs = layer_31 / layer_32
            
            outputs = inputs
        
        return outputs    

def train(ccrf_Y, ccrf_x, t_ccrf_s, g_kernels, epcho, lr, rnn_layer, ccrf_s, T_cuda):
    
    t, n, m = ccrf_x.shape        
    if t_ccrf_s == False:
        M = g_kernels.shape[2]
    else:
        M = g_kernels.shape[2] + 1
    F = (m, 1)
    
    m_crfRnn = CRF_RNN(rnn_layer, F, M, latent=True, kernel_size=(g_kernels.shape[0], g_kernels.shape[0]))
    
    m_loss = torch.nn.MSELoss()
    m_optimizer = torch.optim.Adam(m_crfRnn.parameters(), lr=lr)
    t_loss = np.inf
    threshold = 1e-5
    
    inputs = torch.from_numpy(ccrf_x).float()
    targets = torch.from_numpy(ccrf_Y.T).float().reshape([t, n, 1])
    
    if t_ccrf_s:
        temp_k = np.sum(ccrf_s, axis=0).reshape([n, n, 1]) / t
        g_kernels = np.append(g_kernels, temp_k, axis=2)
    g_kernels = torch.from_numpy(g_kernels).float()
    
    if T_cuda:
        print("Using GPU:")
        inputs = inputs.cuda()
        targets = targets.cuda()
        g_kernels = g_kernels.cuda()
        m_crfRnn = m_crfRnn.cuda()
    else:
        print("Using CPU:")
    
    for i in range(0, epcho):
        m_optimizer.zero_grad()
        outputs = m_crfRnn.forward(inputs, g_kernels, T_cuda)
        loss = m_loss(outputs, targets)
        loss.backward()
        m_optimizer.step()
        if i%100 == 0:
            print(loss.data[0])
        if np.abs(t_loss - loss.data[0]) > threshold and t_loss > loss.data[0]:
            t_loss = loss.data[0]
        else:
            print('minimized!')
            break
    
    return m_crfRnn

def predict(linear_predicts, model, g_kernels, t_ccrf_s, ccrf_s, T_cuda):
    
    t, n, m = linear_predicts.shape
    linear_predicts = torch.from_numpy(linear_predicts).float()
    if t_ccrf_s:
        temp_k = np.sum(ccrf_s, axis=0).reshape([n, n, 1]) / t
        g_kernels = np.append(g_kernels, temp_k, axis=2)
    g_kernels = torch.from_numpy(g_kernels).float()
    
    if T_cuda:
        linear_predicts = linear_predicts.cuda()
        g_kernels = g_kernels.cuda()
        
    results = model.forward(linear_predicts, g_kernels, T_cuda).cpu()
    results = results.detach().numpy()[:,:,0]
    
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
        