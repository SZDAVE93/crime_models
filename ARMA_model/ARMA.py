# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:45:33 2019

@author: yifei
"""
import numpy as np
import statsmodels.api as sm

class m_ARMA():
    
    def __init__(self, AR_factor, MA_factor):
        
        self.AR_factor = AR_factor
        self.MA_factor = MA_factor
    
    def train(self, seq_x):
        
        n, t = seq_x.shape
        models = []
        for i in range(0, n):
            train_x = seq_x[i, :]
            AR_model = sm.tsa.ARMA(train_x, (self.AR_factor, 
                                             self.MA_factor)).fit(method='css-mle', disp=False)
            models.extend([AR_model])
        return models
    
    def predict_iter(self, eval_x, m_model):
        
        AR_days = m_model[0].k_ar
        MA_days = m_model[0].k_ma
        t, num_regions, n = eval_x.shape
        iter_x = eval_x[0, :, :]
    
        pred_y = np.zeros([num_regions, 1])
        for k in range(0, t):
            pred_y_tmp = np.zeros([num_regions, 1])
            for i in range(0, num_regions):
                ar_params = m_model[i].arparams.reshape([AR_days, 1])
                ma_params = m_model[i].maparams.reshape([MA_days, 1])
                ar_params = np.rot90(np.rot90(ar_params))[:, 0]
                ma_params = np.rot90(np.rot90(ma_params))[:, 0]
                pred_y_tmp[i, :] = np.dot(iter_x[i, :], ar_params) + \
                                   np.dot(iter_x[i, 0:MA_days], ma_params) + \
                                   m_model[i].params[0]
            pred_y = np.append(pred_y, pred_y_tmp, axis=1)
            iter_x = np.append(iter_x[:, 1:n], pred_y_tmp, axis=1)
        pred_y = pred_y[:, 1:t+1]
        return pred_y
    
    def predict(self, eval_x, m_model):
        
        AR_days = m_model[0].k_ar
        MA_days = m_model[0].k_ma
        t, num_regions, n = eval_x.shape
    
        pred_y = np.zeros([num_regions, t])
        for i in range(0, num_regions):
            ar_params = m_model[i].arparams.reshape([AR_days, 1])
            ma_params = m_model[i].maparams.reshape([MA_days, 1])
            ar_params = np.rot90(np.rot90(ar_params))[:, 0]
            ma_params = np.rot90(np.rot90(ma_params))[:, 0]
            pred_y[i, :] = np.dot(eval_x[:, i, :], ar_params) + \
                           np.dot(eval_x[:, i, 0:MA_days], ma_params) + \
                           m_model[i].params[0]
        return pred_y