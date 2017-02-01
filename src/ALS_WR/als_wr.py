#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:35:18 2017

@author: Evariste
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing


# Add parent directory to python path
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from data_fetching.data_set import DataSet

class ALS_WR():
    
    """
    // How to use //
    
    For demonstration, run this script under spyder directly, choosing 
    which dataset in __main__ by commenting and decommenting
    
    Let traind_df, test_df be two pandas.DataFrame objects with 3 columns [user_id,item_id,rating]

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
        
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
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
        self.time_for_each_iter = []
        self.RMSE_train_after_each_iter = []
        self.RMSE_test_after_each_iter = []


    def fit(self,n_iter=10):
        t0 = time.time()
        print("\nALS-WR begins...")
        print("with r = %d, lmda = %.3f" % (self.r,self.lmda))
        
        # Parallel computation
        pool = multiprocessing.Pool()
        
        for i in range(n_iter):
            print("\n%d-th iteration begins..." % (self.n_iter_carried_out+1))
            t1 = time.time()
            
            print("Updating X...")
            li_Xu = pool.map(self.find_Xu, self.user_id_unique)
            for uid, Xu in li_Xu:
                self.X[uid-1] = Xu
            print("X updated.")
            
            print("Updating Y...")
            li_Yi = pool.map(self.find_Yi, self.item_id_unique)
            for iid, Yi in li_Yi:
                self.Y[iid-1] = Yi
            print("Y updated.")
            
            print("%d-th iteration finished.\nCalculating training/testing RMSE..." % (self.n_iter_carried_out+1))
            t2 = time.time()
            self.pred()
            rmse_test = self.get_testing_RMSE()
            rmse_train = self.get_training_RMSE()
            self.RMSE_test_after_each_iter.append(rmse_test)
            self.RMSE_train_after_each_iter.append(rmse_train)
            delta_t = t2 - t1
            self.time_for_each_iter.append(delta_t)
            
            self.n_iter_carried_out += 1
            print("Time used: %.2f" % delta_t)
            print("Current training RMSE: %.4f" % rmse_train)
            print("Current testing RMSE: %.4f" % rmse_test)
        
        print("\nALS-WR finished.")
        final_testing_rmse = self.RMSE_test_after_each_iter[-1]
        final_training_rmse = self.RMSE_train_after_each_iter[-1]
        t1 = time.time()
        print("Total time used: %.2f" % (t1-t0))
        print("Final training RMSE: %.4f" % final_training_rmse)
        print("Final testing RMSE: %.4f" % final_testing_rmse)
        pool.close()
        pool.terminate()
        pool.join()
        
        
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
            iid = int(row.item_id)
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
            uid = int(row.user_id)
            xu = self.X[uid-1]
            rui = row.rating
            Ai += xu[:,None] * xu[None,:]
            Vi += rui * xu
        return iid, np.linalg.solve(Ai,Vi)
        
    # Predict using current X and Y and update the column "rating_pred" in test_df
    def pred(self):
        self.test_df["rating_pred"] = self.test_df.apply(lambda row: 
            self.X[int(row.user_id-1)].dot(self.Y[int(row.item_id-1)]),axis=1)
        self.train_df["rating_pred"] = self.train_df.apply(lambda row: 
            self.X[int(row.user_id-1)].dot(self.Y[int(row.item_id-1)]),axis=1)
    
    # Compute the testing RMSE with current X and Y w.r.t test_df
    def get_testing_RMSE(self):
        r_pred = self.test_df["rating_pred"]
        r_real = self.test_df["rating"]
        rmse = np.sqrt(((r_pred - r_real)**2).mean())
        return rmse
        
    # Compute the training RMSE with current X and Y w.r.t train_df
    def get_training_RMSE(self):
        r_pred = self.train_df["rating_pred"]
        r_real = self.train_df["rating"]
        rmse = np.sqrt(((r_pred - r_real)**2).mean())
        return rmse

    def plot_RMSE(self):
        plt.plot(range(1,self.n_iter_carried_out+1),self.RMSE_train_after_each_iter, marker='o', label='Training RMSE')
        plt.plot(range(1,self.n_iter_carried_out+1),self.RMSE_test_after_each_iter, marker='v', label='Testing RMSE')
        plt.title('ALS-WR with r=%d and $\lambda$=%.3f' % (self.r,self.lmda))
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.show()
        
    def get_average_time(self):
        return np.average(self.time_for_each_iter)

def perf_weak(dataset="movielens",size="M",r=100,lmda=0.065,n_iter=10):
    print('START')
    print('Fetch Dataset...')
    ds = DataSet(dataset=dataset, size=size)
    print('Dataset fetched')
    print('Split set in training and test sets...')
    train_df, test_df, _ = ds.split_train_test(False)
    print('Data set is split')
    als = ALS_WR(train_df,test_df,r=r,lmda=lmda)
    als.fit(n_iter=n_iter)
    
    print("\n\n########################### Analysis ###########################")
    print("The RMSE curve for dataset=\"%s\", size=\"%s\" is" % (dataset,size))
    als.plot_RMSE()
    print("Number of iteration carried out: %d" % als.n_iter_carried_out)
    print("Average time for each iteration: %.2f" % als.get_average_time())
    print("Final training RMSE: %.4f" % als.RMSE_train_after_each_iter[-1])
    print("Final testing RMSE: %.4f" % als.RMSE_test_after_each_iter[-1])
    print('END')
    

if __name__ == "__main__":
    # MovieLens dataset 100k
    perf_weak(dataset="movielens",size="S",r=10,lmda=0.065,n_iter=20)
    
    # Toy dataset
    # perf_weak(dataset="toy",r=20,lmda=0.065,n_iter=20)
    
    # MovieLens dataset 1M.
    # Attention: slow, ~200 seconds per iteration
    # perf_weak(dataset="movielens",size="M",r=100,lmda=0.065,n_iter=10)
    
    # Jester dataset
    # Attention: very slow, ~400 seconds per iteration
    #perf_weak(dataset="jester",r=100,lmda=0.065,n_iter=10)
    
            
    

