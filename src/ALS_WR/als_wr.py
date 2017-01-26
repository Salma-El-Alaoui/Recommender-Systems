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

def ALS_WR(df_train):
    print("Beginning...")
    
    # Hyperparameters
    # r : number of latent features
    # lmda : penalization coefficient
    r = 10
    lmda = 0.065
    
    n_users = df_train.user_id.unique().size
    n_items = df_train.item_id.unique().size
    
    X = np.random.rand(n_users,r)
    Y = np.random.rand(n_items,r)
    
    # Input
    #   Y: feature matrix of movies
    #   Du: pd.DataFrame corresponding to u
    # Output
    #   the feature vector x_u for user u
    def findXu(Y,Du):
        nu = Du.shape[0]
        Au = nu * lmda * np.eye(r)
        Vu = np.zeros(r)
        for index, row in Du.iterrows():
            yi = Y[row.item_id-1]
            rui = row.rating
            Au += yi[:,None] * yi[None,:]
            Vu += rui * yi
        return np.linalg.solve(Au,Vu)
        
    # Input
    #   X: feature matrix of movies
    #   Di: pd.DataFrame corresponding to i
    # Output
    #   the feature vector y_i for user i
    def findYi(X,Di):
        ni = Di.shape[0]
        Ai = ni * lmda * np.eye(r)
        Vi = np.zeros(r)
        for index, row in Di.iterrows():
            xu = X[row.user_id-1]
            rui = row.rating
            Ai += xu[:,None] * xu[None,:]
            Vi += rui * xu
        return np.linalg.solve(Ai,Vi)
    
    print("Begin groupby")
    grouped_by_userid = df_train.groupby(['user_id'])
    print("2th groupby")
    grouped_by_itemid = df_train.groupby(['item_id'])
    print("Begin iteration")
    
    for _ in range(10):
        t1 = time.time()
        for user in range(n_users):
            uid = user + 1
            Du = grouped_by_userid.get_group(uid)
            X[user] = findXu(Y,Du)
        for item in range(n_items):
            iid = item + 1
            Di = grouped_by_itemid.get_group(iid)
            Y[item] = findYi(X,Di)
        t2 = time.time()
        print(str(_)+"-th iteration, time: "+str(t2-t1))
    return X, Y
    






#if __name__ == "__main__":
#    print("haha")
