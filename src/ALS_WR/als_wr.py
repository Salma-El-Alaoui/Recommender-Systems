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
from multiprocessing import Pool

# Add parent directory to python path
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from data_fetching.data_set import DataSet

class ALS_WR():
    
    """
    // How to use //
    
    Let traind_df, test_df be two pandas.DataFrame objects with 3 columns [user_id,item_id,raing]

    Initialize the class with:
        
        als = ALS_WR(train_df,test_df,r=100,lmda=0.065)
        
    where r is the latent dimension and lmda the penalization coefficient.

    Begin the ALS iterations with:
        
        als.fit(n_iter=10)
        
    where n_iter is the number of iterations. When doing ALS iteration, the ALS_WR object
    store information like current RMSE and time used for this iteration.
    
    The next time when one does als.fit(), the iteration will use current 
    X and Y after iterations already done. This avoids repeating iterations.
    """
    
    def __init__(self,train_df,test_df,r=100,lmda=0.065):
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
        print("Number of users in the dataset: n_users = %d" % self.n_users)
        print("Number of items in the dataset: n_items = %d" % self.n_items)
        
        # variables memorizing the information after each iteration
        self.n_iter_carried_out = 0
        self.RMSE_test_after_each_iter = []
        self.time_for_each_iter = []


    def fit(self,n_iter=10):
        t0 = time.time()
        print("ALS-WR begins...")
        print("with r = %d, lmda = %.3f" % (self.r,self.lmda))
        rmse = self.get_RMSE()
        print("Initial RMSE: %.4f" % rmse)
        
        for i in range(n_iter):
            print("\n%d-th iteration begins... \nUpdating X..." % (self.n_iter_carried_out+1))
            t1 = time.time()
            
            # Parallel computation
            pool = Pool()
            
            li_Xu = pool.map(self.find_Xu, self.user_id_unique)
            for uid, Xu in li_Xu:
                self.X[uid-1] = Xu
            print("X updated.")
            
            print("Updating Y...")
            li_Yi = pool.map(self.find_Yi, self.item_id_unique)
            for iid, Yi in li_Yi:
                self.Y[iid-1] = Yi
            print("Y updated.")
                
            t2 = time.time()
            delta_t = t2 - t1
            self.time_for_each_iter.append(delta_t)
            print("%d-th iteration finished." % (self.n_iter_carried_out+1))
            self.n_iter_carried_out += 1
            print("Time used: %.2f" % delta_t)
            self.pred()
            rmse = self.get_RMSE()
            self.RMSE_test_after_each_iter.append(rmse)
            print("Current RMSE: %.4f" % rmse)
        
        final_rmse = self.RMSE_test_after_each_iter[-1]
        print("ALS-WR finished.")
        t1 = time.time()
        print("Total time used: %.2f " (t1-t0))
        print("Final RMSE: %.4f" % final_rmse)
            
        
    # Input
    #   uid: the user id of the latent feature vector to be updated
    # Output
    #   uid, the feature vector
    def find_Xu(self,uid):
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
        return uid, np.linalg.solve(Au,Vu)
        
    # Input
    #   iid: the item id of the latent feature vector to be updated
    # Output
    #   iid, the feature vector
    def find_Yi(self,iid):
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
        return iid, np.linalg.solve(Ai,Vi)
        
    # Predict using current X and Y and update the column "rating_pred" in test_df
    def pred(self):
        self.test_df["rating_pred"] = self.test_df.apply(lambda row: 
            self.X[int(row.user_id-1)].dot(self.Y[int(row.item_id-1)]),axis=1)
    
    # Compute the RMSE with current X and Y w.r.t test_df
    def get_RMSE(self):
        r_pred = self.test_df["rating_pred"]
        r_real = self.test_df["rating"]
        rmse = np.sqrt(((r_pred - r_real)**2).mean())
        return rmse