#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:35:18 2017

@author: Evariste
"""

import numpy as np
import pandas as pd
import os
import sys
import time

# Add parent directory to python path
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from data_fetching.data_set import DataSet

class ALS_WR():
    
    def __init__(self,train_df,test_df,r=10,lmda=0.065):
        # Hyperparameters
        # r : number of latent features
        # lmda : penalization coefficient
        self.r = r
        self.lmda = lmda
        
        self.train_df = train_df
        self.test_df = test_df.copy()
        self.test_df["rating_pred"] = 0
        self.n_users = max(train_df.user_id.max(),test_df.user_id.max())
        self.n_items = max(train_df.item_id.max(),test_df.item_id.max())
        self.user_id_unique = train_df.user_id.unique()
        self.item_id_unique = train_df.item_id.unique()
        self.grouped_by_userid = self.train_df.groupby(['user_id'])
        self.grouped_by_itemid = self.train_df.groupby(['item_id'])
        self.X = np.random.rand(self.n_users,r)
        self.Y = np.random.rand(self.n_items,r)
        print("n_users: %d" % self.n_users)
        print("n_items: %d" % self.n_items)
        
        # variables memorizing the information after each iteration
        self.n_iter_carried_out = 0
        self.RMSE_test_after_each_iter = []
        self.time_for_each_iter = []


    def fit(self,n_iter=10):
        print("ALS-WR begins...\n")
        for i in range(n_iter):
            t1 = time.time()
            
            # X = argmin...
            for uid in self.user_id_unique:
                self.X[uid-1] = self.__find_Xu(uid)
                
            # Y = argmin...
            for iid in self.item_id_unique:
                self.Y[iid-1] = self.__find_Yi(iid)
                
            t2 = time.time()
            delta_t = t2 - t1
            self.time_for_each_iter.append(delta_t)
            self.n_iter_carried_out += 1
            print("%d-th iteration finished." % self.n_iter_carried_out)
            print("Time used: %.2f" % delta_t)
            self.pred()
            rmse = self.get_RMSE()
            self.RMSE_test_after_each_iter.append(rmse)
            t3 = time.time()
            print("Current RMSE: %.4f" % rmse)
            print("Time used for get_RMSE: %.2f\n" % (t3-t2))
            
        
    # Input
    #   Y: feature matrix of movies
    #   Du: pd.DataFrame corresponding to u
    # Output
    #   the feature vector x_u for user u
    def __find_Xu(self,uid):
        Du = self.grouped_by_userid.get_group(uid)
        nu = Du.shape[0]
        Au = nu * self.lmda * np.eye(self.r)
        Vu = np.zeros(self.r)
        for index, row in Du.iterrows():
            iid = row.item_id
            yi = self.Y[iid-1]
            rui = row.rating
            Au += yi[:,None] * yi[None,:]
            Vu += rui * yi
        return np.linalg.solve(Au,Vu)
        
    # Input
    #   X: feature matrix of movies
    #   Di: pd.DataFrame corresponding to i
    # Output
    #   the feature vector y_i for item i
    def __find_Yi(self,iid):
        Di = self.grouped_by_itemid.get_group(iid)
        ni = Di.shape[0]
        Ai = ni * self.lmda * np.eye(self.r)
        Vi = np.zeros(self.r)
        for index, row in Di.iterrows():
            uid = row.user_id
            xu = self.X[uid-1]
            rui = row.rating
            Ai += xu[:,None] * xu[None,:]
            Vi += rui * xu
        return np.linalg.solve(Ai,Vi)
        
    def pred(self):
        self.test_df["rating_pred"] = self.test_df.apply(lambda row: 
            self.X[int(row.user_id-1)].dot(self.Y[int(row.item_id-1)]),axis=1)
    
    def get_RMSE(self):
        r_pred = self.test_df["rating_pred"]
        r_real = self.test_df["rating"]
        rmse = np.sqrt(((r_pred - r_real)**2).mean())
        return rmse