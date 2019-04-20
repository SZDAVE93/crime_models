# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:26:56 2018

@author: szdave
"""
import numpy as np
import torch
from random import randrange
import torch.nn.init as Init

class TCP:
    
    def __init__(self, data_y, data_x, data_p, data_q, theta=0.01):
        
        self.Y = data_y
        self.X = data_x
        self.P = data_p
        self.Q = data_q
        self.admm_theta = theta
    
    def Init_Parameters(self, dim_t, dim_n, dim_m):
        
        para_weights = np.random.randn(dim_t, dim_m, dim_n)
        para_E = np.random.randn(dim_t, dim_m, dim_n**2)
        para_U = np.random.randn(dim_t, dim_m, dim_n**2)
        para_F = np.random.randn(dim_n, dim_m, dim_t-1)
        para_V = np.random.randn(dim_n, dim_m, dim_t-1)
        
        return para_weights, para_E, para_U, para_F, para_V
    
    def gradient_W_kn(self, W, E, U, F, V, k, n):
        
        # P shape(N, N**2); Q shape(K, K-1)
        p_row, p_col = self.P.shape
        q_row, q_col = self.Q.shape
        P_nT = self.P[n, :].reshape([1, p_col]).T
        Q_kT = self.Q[k, :].reshape([1, q_col]).T
        
        term_1 = 2*(np.sum(self.X[k, n, :] * W[k, :, n]) - self.Y[n, k])*self.X[k, n, :]
        term_2 = self.admm_theta*np.dot((np.dot(W[k, :, :], self.P) - E[k, :, :] + U[k, :, :]), P_nT)[:, 0]
        term_3 = self.admm_theta*np.dot((np.dot(W[:, :, n].T, self.Q) - F[n, :, :] + V[n, :, :]), Q_kT)[:, 0]

        gradient = term_1 + term_2 + term_3
        return gradient
    
    def converge_W(self, W, E, U, F, V, k, n, lr, epcho, threshold):
        
        temp_w = W[k, :, n].copy()
        result_wkn = 0
        for i in range(0, epcho):
            gradient_W_kn = self.gradient_W_kn(W, E, U, F, V, k, n)
            W[k, :, n] = W[k, :, n] - lr*gradient_W_kn
            result_wkn = W[k, :, n]
            if np.sqrt(np.sum((temp_w - W[k, :, n])**2)) < threshold:
                break
            else:
                temp_w = W[k, :, n].copy()
        return result_wkn
    
    def update_E(self, W, U, k):
        
        term = np.dot(W[k, :, :], self.P) + U[k, :, :]
        alpha = 0.5 / self.admm_theta
        
        indexs_1 = np.where(term > alpha)
        indexs_2 = np.where(np.abs(term) <= alpha)
        indexs_3 = np.where(term < - alpha)
        
        term[indexs_1] = term[indexs_1] - alpha
        term[indexs_2] = 0
        term[indexs_3] = term[indexs_3] + alpha
        
        E_k = term
        return E_k
    
    def update_F(self, W, V, n):
        
        term = np.dot(W[:, :, n].T, self.Q) + V[n, :, :]
        alpha = 1 / self.admm_theta
        
        indexs_1 = np.where(term > alpha)
        indexs_2 = np.where(np.abs(term) <= alpha)
        indexs_3 = np.where(term < - alpha)
        
        term[indexs_1] = term[indexs_1] - alpha
        term[indexs_2] = 0
        term[indexs_3] = term[indexs_3] + alpha
        
        F_n = term
        return F_n
    
    def update_U(self, U, W, E, k):
        
        U_k = U[k, :, :] + np.dot(W[k, :, :], self.P) - E[k, :, :]
        
        return U_k
    
    def update_V(self, V, W, F, n):
        
        V_n = V[n, :, :] + np.dot(W[:, :, n].T, self.Q) - F[n, :, :]
        
        return V_n
    
    def measure_loss(self, W, E, U, F, V):
        
        K, N, M = self.X.shape
        term_1 = 0
        term_3 = 0
        term_5 = 0
        for i in range(0, K):
            temp_y = np.sum((np.diag(np.dot(self.X[i, :, :], W[i, :, :])) - self.Y[:, i])**2)
            term_1 = term_1 + temp_y
            term_p = np.dot(W[i, :, :], self.P) - E[i, :, :] + U[i, :, :]
            term_3 = term_3 + np.sqrt(np.sum(term_p**2))**2
        for i in range(0, N):
            term_q = np.dot(W[:, :, i].T, self.Q) - F[i, :, :] + V[i, :, :]
            term_5 = term_5 + np.sqrt(np.sum(term_q**2))**2
        term_3 = term_3 * self.admm_theta * 0.5
        term_5 = term_5 * self.admm_theta * 0.5
        term_2 = np.sum(np.abs(E)) * 0.5
        term_4 = np.sum(np.abs(F))
        
        m_loss = term_1 + term_2 + term_3 + term_4 + term_5
        regulation = m_loss - term_1
        
        return m_loss, regulation
    
    def ADMM_Optimization(self, lr=1e-3, epcho=1000, threshold_wkn=1e-5):
        
        K, N, M = self.X.shape
        W, E, U, F, V = self.Init_Parameters(K, N, M)
        loss = np.inf
        for i in range(0, epcho):
            # randomly pick k, and n from K, and N
            k = randrange(0, K)
            n = randrange(0, N)
            W[k, :, n] = self.converge_W(W, E, U, F, V, k, n, lr, epcho, threshold_wkn)
            E[k, :, :] = self.update_E(W, U, k)
            F[n, :, :] = self.update_F(W, V, n)
            U[k, :, :] = self.update_U(U, W, E, k)
            V[n, :, :] = self.update_V(V, W, F, n)
            m_loss, regulation = self.measure_loss(W, E, U, F, V)
            if m_loss < loss:
                loss = m_loss
                #print("Loss:\t%.5f\tRegulation:\t%.5f" % (m_loss, regulation))
                #print("Loss:\t%.5f" % m_loss)
            else:
                #print("RMSE_loss:\t%.5f" % np.sqrt((m_loss - regulation) / (K * N)))
                print("Minimized!")
                break
        return W
    
    def W_Optimization(self, lag_days, W, lr=1e-2, epcho=50000):
        
        K, N, M = self.X.shape
        m_W = np.zeros([M, N, K])
        for i in range(0, K):
            m_W[:, :, i] = W[i, :, :]
               
        m_model = torch.nn.Linear(lag_days, 1, bias=True)
        Init.constant_(m_model.weight.data, 1e-2)
        m_loss = torch.nn.MSELoss()
        m_optimizer = torch.optim.SGD(m_model.parameters(), lr=lr)
        t_loss = np.inf
        for i in range(0, epcho):
            loss = 0
            m_optimizer.zero_grad()
            for j in range(lag_days, K):
                in_W = torch.from_numpy(m_W[:, :, j-lag_days:j]).float()
                in_X = torch.from_numpy(self.X[j, :, :]).float()
                real_Y = torch.from_numpy(self.Y[:, lag_days].reshape([N, 1])).float()
                out_W = m_model.forward(in_W)
                out_Y = torch.diag(in_X.mm(out_W[:, :, 0])).reshape([N, 1])
                loss = loss + m_loss(out_Y, real_Y)
            loss = torch.div(loss, K-lag_days)
            loss.backward()
            m_optimizer.step()
            if loss.data[0] < t_loss:
                t_loss = loss.data[0]
                #print(loss.data[0])
            else:
                print("Minimized!")
                break
            
        
        return m_model
    
    
