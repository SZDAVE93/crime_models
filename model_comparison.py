# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:46:24 2018

@author: szdave
"""
import os
import torch
import numpy as np
import torch.nn.init as Init
from CRF_RNN_model import CRF_RNN
from TCP_model import TCP_model
from LSTM_model import LSTM_model
from NN_CCRF_model import NN_CCRF
from Linear_model import linear_regression

def method(kernels, train_data, eval_data):
    
    # hyper-parameters
    lag_days = 4
    dim_hidden = 64
    crf_rnn_layer = 6
    
    
    print("Hidden_state:\t%d" % dim_hidden)
    print("SDAE layers:\t%d" % crf_rnn_layer)
    
    train_x = train_data[0]
    train_y = train_data[1]
    train_s = train_data[2]
    eval_x = eval_data[0]
    eval_y = eval_data[1]
    eval_s = eval_data[2]
    
    epcho = 10000
    lr_linear = 1e-2
    lr_crf = 1e-2
    lr_tcp = 1e-3
    lr_lstm = 1e-2
    lr_nncrf = 1e-2
    ccrfs_tag = False
    T_cuda = False
    threshold = 1e-3
    
    # used for co-train in lstm
    t, n, m = train_x.shape
    lstm_x = np.zeros([t, m, n])
    for i in range(t):
        lstm_x[i, :, :] = train_x[i, :, :].T
        
    t, n, m = eval_x.shape
    lstm_eval_x = np.zeros([t, m, n])
    for i in range(t):
        lstm_eval_x[i, :, :] = eval_x[i, :, :].T
    
    '''
    training on different models
    '''
    # learn simple linear model
    print("Learning Linear model:")
    linear_model = linear_regression.train(train_x, train_y, epcho, lr_linear, threshold)
    predicts_linear = linear_regression.predict(train_x, linear_model)
    # learn lstm model
    print("Learning LSTM model:")
    lstm_model = LSTM_model.train(train_x, train_y, dim_hidden, epcho, lr_lstm, threshold)
    predicts_lstm = LSTM_model.predict(lstm_model, train_x)
    # learn tcp model
    print("Learning TCP model:")
    TCP_W, tcp_model = TCP_model.train(lr_tcp, epcho, train_x, train_y, kernels, lag_days)
    
    # learn crf_rnn model
    print("Learning CRFasRNN model:")
    CRFasRNN_model = CRF_RNN.train(train_y, train_x, ccrfs_tag, kernels, epcho, lr_crf, 
                                   crf_rnn_layer, train_s, T_cuda)
    # learn NN-CCRF model
    print("Learning NN-CCRF with Linear:")
    NN_CCRF_linear1 = NN_CCRF.train_m(linear_model, train_x, train_y, crf_rnn_layer, epcho, lr_nncrf, threshold)   
    
    print("Learning NN-CCRF wiht LSTM:")
    NN_CCRF_lstm1 = NN_CCRF.train_m(lstm_model, lstm_x, train_y, crf_rnn_layer, epcho, lr_nncrf, threshold)
    
    
    #predictions on different models
    m = eval_y.shape[1]
    predicts_history = eval_y[:, 0:m-1]
    predicts_linear = linear_regression.predict(eval_x, linear_model).T
    predicts_tcp = TCP_model.Predict(TCP_W, eval_x, tcp_model, lag_days)
    predicts_lstm = LSTM_model.predict(lstm_model, eval_x).T
    predicts_CRFasRNN = CRF_RNN.predict(eval_x, CRFasRNN_model, kernels, ccrfs_tag, train_s, T_cuda).T
    predicts_NN_CCRF_linear1 = NN_CCRF.predict_m(NN_CCRF_linear1, eval_x).T
    predicts_NN_CCRF_lstm1 = NN_CCRF.predict_m(NN_CCRF_lstm1, lstm_eval_x).T
    
    
    #evaluation metrics
    n = eval_y.shape[0]
    time_len = m-1
    base = n * time_len
    rmse_history = np.sqrt(np.sum((eval_y[:, 1:m] - predicts_history)**2)/base)
    rmse_linear = np.sqrt(np.sum((eval_y[:, 1:m] - predicts_linear)**2)/base)
    rmse_tcp = np.sqrt(np.sum((eval_y[:, 1:m] - predicts_tcp)**2)/base)
    rmse_lstm = np.sqrt(np.sum((eval_y[:, 1:m] - predicts_lstm)**2)/base)
    rmse_CRFasRNN = np.sqrt(np.sum((eval_y[:, 1:m] - predicts_CRFasRNN)**2)/base)
    rmse_NN_CCRF_linear1 = np.sqrt(np.sum((eval_y[:, 1:m] - predicts_NN_CCRF_linear1)**2)/base)
    rmse_NN_CCRF_lstm1 = np.sqrt(np.sum((eval_y[:, 1:m] - predicts_NN_CCRF_lstm1)**2)/base)
    
    
    print("RMSE results:")
    print("History:\t\t%.5f" % rmse_history)
    print("Linear:\t\t\t%.5f" % rmse_linear)
    print("LSTM:\t\t\t%.5f" % rmse_lstm)
    print("CRFasRNN:\t\t%.5f" % rmse_CRFasRNN)
    print("TPC:\t\t\t%.5f" % rmse_tcp)
    print("NN-CCRF_linear:\t\t%.5f" % rmse_NN_CCRF_linear1)
    print("NN-CCRF_lstm:\t\t%.5f" % rmse_NN_CCRF_lstm1)
    
if __name__ == "__main__":
    
    
    # two types of crime type, person or property
    # two different city CHI: chicago, NY: New York
    crime_type = ["person", "property"]
    city_name = ["NY", "CHI"]
    path = os.getcwd()
    
    crime_ID = 0
    city_ID = 0
    
    kernels = np.load(path + '/data/' + city_name[city_ID] + '/kernels.npy')
    train_x = np.load(path + '/data/' + city_name[city_ID] + '/'+ crime_type[crime_ID] + '_train_x.npy')
    train_y = np.load(path + '/data/' + city_name[city_ID] + '/'+ crime_type[crime_ID] + '_train_y.npy')
    train_s = np.load(path + '/data/' + city_name[city_ID] + '/'+ crime_type[crime_ID] + '_train_s.npy')
    eval_x = np.load(path + '/data/' + city_name[city_ID] + '/'+ crime_type[crime_ID] + '_eval_x.npy')
    eval_y = np.load(path + '/data/' + city_name[city_ID] + '/'+ crime_type[crime_ID] + '_eval_y.npy')
    eval_s = np.load(path + '/data/' + city_name[city_ID] + '/'+ crime_type[crime_ID] + '_eval_s.npy')

    
    print("CITY:\t%s\tCRIME_TYPE:\t%s" %(city_name[city_ID], crime_type[crime_ID]))
    method(kernels, (train_x, train_y, train_s), (eval_x, eval_y, eval_s))
    
    
    
    