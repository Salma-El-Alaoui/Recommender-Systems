import io
import csv
import pandas as pd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

def getDataframe(dataset = 'movielens', size = 'S'):
    """
    @Parameters:
    ------------
    dataset: String    -- 'movielens' or 'netflix'
    size:    String    -- 'S', 'M' or 'L'
    
    @Return:
    --------
    df:      DataFrame -- columns = UserId || ItemId || Rating || Timestamp
    
    @Infos:
    -------
    For movielens:
        -> Size = S: 100K ratings,  1K users, 1.7K movies, ~   2MB
        -> Size = M:   1M ratings,  6K users,   4K movies, ~  25MB
        -> Size = L:  10M ratings, 72K users,  10K movies, ~ 265MB
            
        All users have rated at least 20 movies no matter the size of the dataset
    """    
    url_map = {
        'S': "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
        'M': "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        'L': "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    }
    
    filename_map = {
        'S': "ml-100k/u.data",
        'M': "ml-1m/ratings.dat",
        'L': "ml-10m/ratings.dat"
    }
    
    separator_map = {
        'S': '\t',
        'M': '::',
        'L': '::'
    }
    
    url = urlopen(url_map[size])
    zipfile = ZipFile(BytesIO(url.read()))
    unzipfile  = io.TextIOWrapper(zipfile.open(filename_map[size], 'r'))
    df = pd.read_csv(unzipfile, sep=separator_map[size], header=None)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    
    return df

def fromDFtoDenseMatrix(df):
    """
    @Parameters:
    ------------
    df:    DataFrame -- columns = UserId || ItemId || Rating || Timestamp
    
    @Return:
    --------
    res:   Dense nparray, 
           shape = (# user_id, # item_id), 
           element[i][j] = rating for user_id[i], item_id[j]  if rating exists
                           nan.                               otherwise
    """
    user_id_max = np.max(df.user_id)
    item_id_max = np.max(df.item_id)
    res = np.nan*np.zeros((user_id_max + 1, item_id_max + 1))
    for row in df.values:
        res[row[0]][row[1]] = row[2]
    return res

def getDataframes_CV():
    """
    @Parameters:
    ------------
    None for now...
    
    
    @Return:
    --------
    folds: Array of 5 dictionaries (5 folds of the CV). Each dictionary has two entries: 'base' and 'test'. 
           'base' corresponds to the training set and 'test' to the test set.
           
    @Infos:
    -------
    The set used for this function is the small one from movielens: 
        100K ratings, 1K users, 1.7K movies, ~ 2MB
    """
    folds = []
    url = urlopen("http://files.grouplens.org/datasets/movielens/ml-100k.zip")
    zipfile = ZipFile(BytesIO(url.read()))
    for i in range(1, 6): 
        train_test = {}
        for ext in ['base', 'test']:
            filename = 'ml-100k/u' + str(i) + '.' + ext
            unzipfile  = io.TextIOWrapper(zipfile.open(filename, 'r'))
            df = pd.read_csv(unzipfile, sep='\t', header=None)
            df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
            train_test[ext] = df
        folds.append(train_test)
    return folds