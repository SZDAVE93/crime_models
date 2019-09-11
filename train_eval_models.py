# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:25:52 2019

@author: yifei
"""

import os
import time
import argparse
import numpy as np
from DataBuilder import d_Builder
from ModelBuilder import m_Builder


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Linear',
                        help='model name (default: Linear)')
    parser.add_argument('--train_end', type=int, default=760,
                        help='end date of training dataset (default: 760)')
    parser.add_argument('--train_days', type=int, default=180,
                        help='num of train days (default: 180)')
    parser.add_argument('--eval_days', type=int, default=30,
                        help='num of eval days (default: 30), but this is for one day prediction')
    parser.add_argument('--ar_days', type=int, default=7,
                        help='historical days in prediction (default: 7)')
    parser.add_argument('--kernel_names', type=str, default='dis', nargs='+',
                        help='kernel type for inference (default: dis)')
    parser.add_argument('--simi_len', type=int, default=30,
                        help='days for building dis kernel (default: 30)')
    parser.add_argument('--data_path', type=str, 
                        default="D:/yifei/Documents/Codes_on_GitHub/External_data/CHI_Region/",
                        help='data path')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=1e-3)
    parser.add_argument('--hyper_parameters', type=int, nargs='+')
    
    args = parser.parse_args()
    train_end = args.train_end
    train_days = args.train_days
    eval_days = args.eval_days
    ar_days = args.ar_days
    data_path = args.data_path
    model_name = args.model_name
    kernel_names = args.kernel_names
    simi_len = args.simi_len
    learning_rate = args.learning_rate
    iters = args.iters
    threshold = args.threshold
    hyper_parameters = args.hyper_parameters
    if model_name == 'NN-CCRF':
        is_nnccrf = True
    else:
        is_nnccrf = False
    if model_name == 'ARMA':
        ar_days = hyper_parameters[0]
    # load training and evaluation dataset
    m_dataBuilder = d_Builder(data_path, ar_days, 
                 train_end, train_days, eval_days)
    (train_x, train_y), (eval_x, eval_y) = m_dataBuilder.load_xy(is_nnccrf)
    if model_name == 'TCP':
        tcp_kernel = m_dataBuilder.load_tcp_kernel(kernel_names)
    if model_name == 'CRFasRNN':
        crf_kernels = m_dataBuilder.load_kernels(kernel_names, simi_len)
    
    # Train the model
    m_modelBuilder = m_Builder(model_name, learning_rate, iters, threshold)
    if model_name == 'TCP':
        hyper_parameters = [hyper_parameters]
        hyper_parameters.extend([tcp_kernel])
    if model_name == 'CRFasRNN':
        hyper_parameters = [hyper_parameters]
        hyper_parameters.extend([crf_kernels])
    print(hyper_parameters)
    m_modelBuilder.train_model(hyper_parameters, train_x, train_y)
    rmse, rmse_std = m_modelBuilder.eval_model(eval_x, eval_y)
    print('{} prediction performance:'.format(model_name))
    print('RMSE:\t{:.5f}\tRMSE_STD:\t{:.5f}'.format(rmse, rmse_std))
    if rmse != -1 and rmse is not None:
        if model_name == 'TCP' or model_name == 'CRFasRNN':
            hyper_parameters = hyper_parameters[0]
        file = open('{}results_{}.txt'.format(data_path, model_name), 'a+')
        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.5f}\t{:.5f}\n'.format(train_end, 
                   train_days, eval_days, ar_days, simi_len, kernel_names, learning_rate, 
                   hyper_parameters, iters, rmse, rmse_std))
        file.close()
    
    
    
    
    
    
    
    
    
