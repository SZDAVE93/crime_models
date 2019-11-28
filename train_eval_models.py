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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Linear',
                        help='model name (default: Linear)')
    parser.add_argument('--train_end', type=int, default=760,
                        help='end date of training dataset (default: 760)')
    parser.add_argument('--train_days', type=int, default=180,
                        help='num of train days (default: 180)')
    parser.add_argument('--eval_days', type=int, default=21,
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
    rmse, rmse_std, pred_y, true_y = m_modelBuilder.eval_model(eval_x, eval_y)
    
    # rmse results
    n_regions = pred_y.shape[0]
    error_1 = (pred_y[:, 0:1] - true_y[:, 0:1])**2
    error_7 = (pred_y[:, 0:7] - true_y[:, 0:7])**2
    error_14 = (pred_y[:, 0:14] - true_y[:, 0:14])**2
    error_21 = (pred_y[:, 0:21] - true_y[:, 0:21])**2
    rmse_1 = np.sqrt(np.sum(error_1) / (n_regions*1))
    rmse_7 = np.sqrt(np.sum(error_7) / (n_regions*7))
    rmse_14 = np.sqrt(np.sum(error_14) / (n_regions*14))
    rmse_21 = np.sqrt(np.sum(error_21) / (n_regions*21))
    
    
    # ranking results
    rankings_1 = ranking_MAP(true_y[:, 0:1], pred_y[:, 0:1], 10)
    rankings_7 = ranking_MAP(true_y[:, 0:7], pred_y[:, 0:7], 10)
    rankings_14 = ranking_MAP(true_y[:, 0:14], pred_y[:, 0:14], 10)
    rankings_21 = ranking_MAP(true_y[:, 0:21], pred_y[:, 0:21], 10)
    results = np.zeros([1, 8])
    results[0, 0] = np.mean(rankings_1[0:5])
    results[0, 1] = np.mean(rankings_1[0:10])
    results[0, 2] = np.mean(rankings_7[0:5])
    results[0, 3] = np.mean(rankings_7[0:10])
    results[0, 4] = np.mean(rankings_14[0:5])
    results[0, 5] = np.mean(rankings_14[0:10])
    results[0, 6] = np.mean(rankings_21[0:5])
    results[0, 7] = np.mean(rankings_21[0:10])
    
    print('{} prediction performance:'.format(model_name))
    print('RMSE:\t{:.5f}\tRMSE_STD:\t{:.5f}'.format(rmse, rmse_std))
    if rmse != -1 and rmse is not None:
        if model_name == 'TCP' or model_name == 'CRFasRNN':
            hyper_parameters = hyper_parameters[0]
        file = open('{}results_{}.txt'.format(data_path, model_name), 'a+')
        file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t'.format(train_end, 
                   train_days, eval_days, ar_days, simi_len, kernel_names, learning_rate, 
                   hyper_parameters, iters))
        file.write('{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t'.format(rmse_1, rmse_7, rmse_14, rmse_21))
        file.write('{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n'.format(
                    results[0, 0], results[0, 1], results[0, 2], results[0, 3],
                    results[0, 4], results[0, 5], results[0, 6], results[0, 7]))
        file.close()
    
    
    
    
    
    
    
    
    
