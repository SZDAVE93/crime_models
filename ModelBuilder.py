# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:16:37 2019

@author: yifei
"""
import numpy as np
from ARMA_model import ARMA
from CRF_RNN_model import CRF_RNN
from TCP_model import TCP_model
from LSTM_model import LSTM_model
from NN_CCRF_model import NN_CCRF
from Linear_model import linear_regression

class m_Builder():
    
    def __init__(self, model_name, learning_rate, iters, threshold):
        
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.iters = iters
        self.threshold = threshold
    
    def train_model(self, hyper_parameters, x, y):
        
        self.model = []
        
        if self.model_name == 'ARMA':
            
            t_model = ARMA.m_ARMA(hyper_parameters[0], hyper_parameters[1])
            m_model = t_model.train(y)
            self.model.extend([m_model])
            self.model.extend([t_model])
        
        if self.model_name == 'Linear':
            m_model = linear_regression.train_global(train_x = x,
                                                   train_y = y,
                                                   lr = self.learning_rate,
                                                   iters = self.iters,
                                                   threshold = self.threshold)
            self.model.extend([m_model])
            
        if self.model_name == 'TCP':
            if hyper_parameters[0][0] == 0:
                print('please input hyper-parameters of the model:\n1.lag_days\t2.lambda\t3.theta')
            else:
                m_model = TCP_model.train(train_x = x, train_y = y, 
                                            lr = self.learning_rate,
                                            iters = self.iters,
                                            threshold = self.threshold,
                                            tcp_kernel = hyper_parameters[1],
                                            lag_days = hyper_parameters[0][0],
                                            lamda = hyper_parameters[0][1],
                                            theta = hyper_parameters[0][2])
                self.model.extend([m_model])
            
        if self.model_name == 'LSTM':
            if hyper_parameters == 0:
                print('please input hyper-parameters of the model:\n1.dim_hidden\t2.num_layers')
            else:
                m_model = LSTM_model.train(train_x = x, train_y = y, 
                                           lr = self.learning_rate,
                                           iters = self.iters,
                                           threshold = self.threshold,
                                           dim_hidden = hyper_parameters[0],
                                           num_layers = hyper_parameters[1])
                self.model.extend([m_model])
            
        if self.model_name == 'CRFasRNN':
            if len(hyper_parameters) != 2:
                print('please input hyper-parameters of the model:\n1.rnn_layers')
            else:
                m_model = CRF_RNN.train(train_x = x, train_y = y, 
                                        lr = self.learning_rate,
                                        iters = self.iters,
                                        threshold = self.threshold,
                                        crf_kernels = hyper_parameters[1],
                                        rnn_layers = hyper_parameters[0][0])
                self.model.extend([m_model])
            
        if self.model_name == 'NN-CCRF':
            if len(hyper_parameters) != 2:
                print('please input hyper-parameters of the model:\n1.dim_hidden\t2.rnn_layers')
            else:
                lstm = LSTM_model.train(train_x = x.transpose(0, 2, 1),
                                        train_y = y, 
                                        lr = self.learning_rate,
                                        iters = 10,
                                        threshold = self.threshold,
                                        dim_hidden = hyper_parameters[0],
                                        num_layers = 1)
                m_model = NN_CCRF.train_m(train_x = x.transpose(1, 0, 2), 
                                          train_y = y,
                                          lr = self.learning_rate,
                                          iters = self.iters,
                                          threshold = self.threshold,
                                          feature_model = lstm,
                                          rnn_layers = hyper_parameters[1])
                self.model.extend([m_model])
    
    def eval_model(self, x, y):
        
        if len(self.model) >= 1:
            m_model = self.model[0]
            n, m = y.shape
            pred_history = y[:, 0:m-1]
            true_y = y[:, 1:m]
            num = n * (m-1)
            if self.model_name == 'TCP':
                pred_y = TCP_model.predict_iter(x, m_model, len(list(m_model[1].parameters())[0][0]))
            elif self.model_name == 'Linear':
                pred_y = m_model.predict_iter(x, m_model)
            elif self.model_name == 'ARMA':
                pred_y = self.model[1].predict_iter(x, m_model)
            elif self.model_name == 'CRFasRNN' or self.model_name == 'LSTM':
                pred_y = m_model.predict_iter(x, m_model)
            elif self.model_name == 'NN-CCRF':
                pred_y = m_model.predict_iter(x, m_model)
            error = (pred_y - true_y)**2
            overall_rmse = np.sqrt(np.sum(error) / num)
            overall_std = np.std(np.sqrt(np.sum(error, axis=0) / n))
            h_error = (pred_history - true_y)**2
            h_overall_rmse = np.sqrt(np.sum(h_error) / num)
            h_overall_std = np.std(np.sqrt(np.sum(h_error, axis=0) / n))
            print('Data original std:\t{:.5f}'.format(np.std(true_y)))
            print('Historical prediction performance:')
            print('RMSE:\t{:.5f}\tRMSE_STD:\t{:.5f}'.format(h_overall_rmse, h_overall_std))
        else:
            print('model error:\twe dont have a model!')
            overall_rmse = -1
            overall_std = -1
        
        return overall_rmse, overall_std, pred_y, true_y
        
            
            
            
    
    