# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:37:57 2017

@author: Sophia
"""


"""
This class implements Probabilistic Matrix factorization with fixed priors (PMF), described in:
    Ruslan Salakhutdinov and Andriy Mnih Department of Computer Science, University of .
    "Bayesian probabilistic matrix factorization using markov chain monte carlo"
    Proceedings of the 25th international conference on Machine learning, ACM, 2008, pp. 880.
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import *
import os
import sys
import time

# Add parent directory to python path
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from data_fetching.data_set import DataSet

class PMF(object):
    def __init__(self, num_latent_feat=20, learning_rate=1, _lambda=0.1, momentum=0.95, maxepoch=20, num_batches=100,
                 batch_size=1000):
        """
        @Parameters:
        ------------
        U : User profile matrix
        V : Item profile matrix
        num_latent_feat : Number of latent features for U and V
        learning_rate : Learning rate
         _lambda : Ridge penalization (on U and V)
        momentum : Momentum of the gradient,
        maxepoch : Number of epochs before stop
        -->Optimization using SGD:
            num_batches : Number of batches (for each epoch)
            batch_size = batch_size : Number of training samples used in each batches
        """
        
        self.V = None
        self.U = None  
        self.num_latent_feat = num_latent_feat  
        self.learning_rate = learning_rate
        self._lambda = _lambda  
        self.momentum = momentum  
        self.maxepoch = maxepoch  
        self.num_batches = num_batches  
        self.batch_size = batch_size
        #Train/Test 
        self.err_train = []
        self.err_val = []
        self.data = None
        self.train_data = None
        self.test_data = None
        self.train_rmse = []
        self.test_rmse = []


    #Fit the PMF model with SGD as the optimization algorithm
    def fit(self, train_set, test_set):
        t_init=time.time()
        #get the average of ratings to center the data
        self.mean_inv = np.mean(train_set[:, 2]) 
        
        #get the size (number of samples of both training and test sets)
        tr_shape = train_set.shape[0]
        te_shape = test_set.shape[0]  
        
        num_user = int(max(np.amax(train_set[:, 0]), np.amax(test_set[:, 0]))) + 1  
        num_item = int(max(np.amax(train_set[:, 1]), np.amax(test_set[:, 1]))) + 1  

        incremental = False
        if ((not incremental) or (self.V is None)):
            # initialization
            self.epoch = 0
            
            #U and V follow a normal distribution
            self.V = 0.1 * np.random.randn(num_item, self.num_latent_feat)
            self.U = 0.1 * np.random.randn(num_user, self.num_latent_feat)  
            # initialization of the increments of U and V used in SGD optimization
            self.V_inc = np.zeros((num_item, self.num_latent_feat))  
            self.U_inc = np.zeros((num_user, self.num_latent_feat))

        while self.epoch < self.maxepoch:
            #t_i_epoch = time.time()
            self.epoch += 1
            
            # Shuffling for SGD
            shuffled_order = np.arange(train_set.shape[0])
            np.random.shuffle(shuffled_order)  

            # Batch update
            for batch in range(self.num_batches):
                
                t_i_epoch=time.time()
                
                batch_idx = np.mod(np.arange(self.batch_size * batch, self.batch_size * (batch + 1)),
                                   shuffled_order.shape[0])  
                batch_invID = np.array(train_set[shuffled_order[batch_idx], 0], dtype='int32')
                batch_comID = np.array(train_set[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.U[batch_invID, :], self.V[batch_comID, :]),
                                  axis=1)  # mean_inv subtracted
                rawErr = pred_out - train_set[shuffled_order[batch_idx], 2] + self.mean_inv

                # Grad_i
                Ix_V = 2 * np.multiply(rawErr[:, np.newaxis], self.U[batch_invID, :]) + self._lambda * self.V[
                                                                                                         batch_comID, :]
                Ix_U = 2 * np.multiply(rawErr[:, np.newaxis], self.V[batch_comID, :]) + self._lambda * self.U[
                                                                                                         batch_invID, :]

                dV = np.zeros((num_item, self.num_latent_feat))
                dU = np.zeros((num_user, self.num_latent_feat))

                for i in range(self.batch_size):
                    dV[batch_comID[i], :] += Ix_V[i, :]
                    dU[batch_invID[i], :] += Ix_U[i, :]

                # Update with momentum
                self.V_inc = self.momentum * self.V_inc + self.learning_rate * dV / self.batch_size
                self.U_inc = self.momentum * self.U_inc + self.learning_rate * dU / self.batch_size
                self.V = self.V - self.V_inc
                self.U = self.U - self.U_inc

                # Compute Objective Function after the update of the momentum
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.U[np.array(train_set[:, 0], dtype='int32'), :],
                                                  self.V[np.array(train_set[:, 1], dtype='int32'), :]),
                                      axis=1)
                    rawErr = pred_out - train_set[:, 2] + self.mean_inv
                    obj = LA.norm(rawErr) ** 2 + 0.5 * self._lambda * (LA.norm(self.U) ** 2 + LA.norm(self.V) ** 2)
                    self.err_train.append(np.sqrt(obj / tr_shape))
                
                # Compute RMSE
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.U[np.array(test_set[:, 0], dtype='int32'), :],
                                                  self.V[np.array(test_set[:, 1], dtype='int32'), :]),
                                      axis=1)  
                    rawErr = pred_out - test_set[:, 2] + self.mean_inv
                    
                    #RMSE
                    self.err_val.append(LA.norm(rawErr) / np.sqrt(te_shape))

                    # RMSE prints
                if batch == self.num_batches - 1:
                    print('Training RMSE: %f, Test RMSE %f' % (self.err_train[-1], self.err_val[-1]))
                    self.train_rmse.append(self.err_train[-1])
                    self.test_rmse.append(self.err_val[-1])
                    t_f_epoch=time.time()
                    print("Time used for this epoch: %s" % (t_f_epoch-t_i_epoch))
                
            
        t_final=time.time()
        print("Time excecution: %s" % (t_final-t_init))
                    
        #Retrieve the best RMSE
        print('RMSE min')
        min_rmse = min(self.test_rmse)
        print('Test set minimal RMSE %f' % min_rmse)
        return self.train_rmse[-1],self.test_rmse[-1]
        
        
                    
    #Predict rating of item for user
    def predict(self, invID):
        return np.dot(self.V, self.U[int(invID), :]) + self.mean_inv  
        
        
#-------------------------------------------------------------------------------------------------------------------
#Main
"""
In order to select which dataset to test, only decomment one of the following lines and select either S or M 
for movielens:

    #ds = DataSet(dataset='movielens', size ='S')
    #ds = DataSet(dataset='jester')
    #ds = DataSet(dataset='toy')

The parameters can be modified either on _init or using pmf(...)
"""
if __name__ == "__main__":
    print('START')
    #Compute the RMSE by epoch as well as the excecution and epoch times
    print('Compute the RMSE by epoch as well as the excecution and epoch times...')
    pmf = PMF()
    print('Fetch dataset...')
    ds = DataSet(dataset='movielens', size ='S')
    #ds = DataSet(dataset='toy')
    print('Dataset fetched')
    ds_v= ds.get_df()
    ratings = ds_v.values
    print('Get dataset description...')
    print(ds.get_description())
    #print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_latent_feat)
    print("Split set in training and test set...")
    train_init, test_init, _ = ds.split_train_test(strong_generalization=False, train_size = 0.8)
    print("Set splitted")
    train = train_init.values
    test = test_init.values
    print('Fit the model...')
    pmf.fit(train, test)
    print('Model fitted')
    print('END')
    
    # Plot train and test errors
    #plt.plot(range(pmf.maxepoch), pmf.train_rmse, marker='o', label='Training Data')
    #plt.plot(range(pmf.maxepoch), pmf.test_rmse, marker='v', label='Test Data')
    #plt.title('Learning Curve')
    #plt.xlabel('Number of Epochs')
    #plt.ylabel('RMSE')
    #plt.legend()
    #plt.grid()
    #plt.show()
    
    
    """
    @Impact of parameters on the performance of the algorithme:
    Please uncomment each of the following to visualize the result
        Plot 1 : Impact of the number of latent features
        Plot 2 : Selection of teh suitable number of latent features given all other parameters fixed
        Plot 3 : Impact of the batch size
    
    ----------------------------------------------------------------
    #Plot RMSE according to Number of latent features to show the sensitivity of PMF to the number of latent features
    D= range(10,50,2)
    R=[]
    for i in D:
        pmf = PMF(num_latent_feat=i)
        res = pmf.fit(train, test)
        R.append(res)
    print(R)
    plt.scatter(D,R)
    plt.title('RMSE by latent features number')
    plt.xlabel('Number of Latent Features')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    ----------------------------------------------------------------
    
    ----------------------------------------------------------------
    #Plot RMSE according to Number of latent features on both train and test sets to select the best value given
    #all the other parameters fixed.
    D= range(5,50,3)
    R=[]
    J=[]
    for i in D:
        pmf = PMF(num_latent_feat=i)
        r,j = pmf.fit(train, test)
        R.append(r)
        J.append(j)
    print(R)
    plt.scatter(D,R, pmf.train_rmse, marker='o', label='Training Data')
    plt.scatter(D,J,pmf.train_rmse, marker='v', label='Test Data')
    plt.title('RMSE by latent features number')
    plt.xlabel('Number of Latent Features')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    ----------------------------------------------------------------
    
    ----------------------------------------------------------------
    #Plot RMSE according to the batch size on train and test sets to select the best value given
    #all the other parameters fixed.
    D= [10, 100, 300, 500, 1000, 3000, 5000]
    R=[]
    J=[]
    for i in D:
        pmf = PMF(batch_size=i)
        r,j = pmf.fit(train, test)
        R.append(r)
        J.append(j)
    print(R)
    plt.scatter(D,R, pmf.train_rmse, marker='o', label='Training Data')
    plt.scatter(D,J,pmf.train_rmse, marker='v', label='Test Data')
    plt.title('RMSE by latent features number')
    plt.xlabel('Number of Latent Features')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    ----------------------------------------------------------------
    """
    

