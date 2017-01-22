import io
import csv
import pandas as pd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

USER_ID = 'user_id'
ITEM_ID = 'item_id'
RATING = 'rating'
TIMESTAMP = 'timestamp'

def getDataframe(dataset = 'movielens', size = 'S'):
    """
    @Parameters:
    ------------
    dataset: String    -- 'movielens' or 'jester'
    size:    String    -- 'S', 'M' or 'L' (only for 'movielens')
    
    @Return:
    --------
    df:      DataFrame -- columns = UserId || ItemId || Rating
    
    @Infos:
    -------
    For movielens:
        -> Size = S:   100K ratings,  1K users, 1.7K movies, ~   2MB, scale: [ 1  , 5 ], density: 
        -> Size = M:     1M ratings,  6K users,   4K movies, ~  25MB, scale: [ 1  , 5 ], density: 4.26%
        -> Size = L:    10M ratings, 72K users,  10K movies, ~ 265MB, scale: [0.5 , 5 ], density: 0.52%
      
    For Jester:
        -> Uniq. size: 1.7M ratings, 60K users,  150  jokes, ~  33MB, scale: [-10 , 10], density: 31.5%
           Values are continuous.
    
        All users have rated at least 20 movies no matter the size of the dataset
    """    
    
    # Define some constants
    DATASETS = ['movielens', 'jester']
    DATASET_WITH_SIZE = ['movielens'] 
    SIZE = ['S', 'M', 'L']
        
    # Check inputs
    if dataset not in DATASETS:
        raise NameError("This dataset is not allowed.")
    if size not in SIZE and dataset in DATASET_WITH_SIZE:
        raise NameError("This size is not allowed.")
    
    # Change size if necessary
    size = size if dataset in DATASET_WITH_SIZE else 'unique'
    
    # Configure url, filename, separator and columns in csv
    if dataset == 'movielens':
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
        
        columns_map = {
            'S': [USER_ID, ITEM_ID, RATING, TIMESTAMP],
            'M': [USER_ID, ITEM_ID, RATING, TIMESTAMP],
            'L': [USER_ID, ITEM_ID, RATING, TIMESTAMP]
        }
        
    if dataset == 'jester':
        url_map = {
            'unique': "http://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip",
        }

        filename_map = {
            'unique': "jester_ratings.dat",
        }
        
        separator_map = {
            'unique': '\t\t'
        }

        columns_map = {
            'unique': [USER_ID, ITEM_ID, RATING],
        }
                
    # Load data in memory
    url = urlopen(url_map[size])
    zipfile = ZipFile(BytesIO(url.read()))
    unzipfile  = io.TextIOWrapper(zipfile.open(filename_map[size], 'r'))
    df = pd.read_csv(unzipfile, sep=separator_map[size], header=None)
    df.columns = columns_map[size]
    df = df[[USER_ID, ITEM_ID, RATING]]
    
    return df

def getDataframe_toy(u=100, i=1000, u_unique=10, i_unique=5, density=0.1, noise=0.3, score_low=0, score_high=5, out='dataframe'):
    """
    @Parameters:
    ------------
    u:           Integer   -- Number of users
    i:           Integer   -- Number of items
    u_unique:    Integer   -- Number of user's type
    i_unique:    Integer   -- Number of item's type 
    density:     Float     -- Percentage of non-nan values
    noise:       Float     -- Each rating is r_hat(i,j) = r(i,j) + N(0, noise) where N is the Gaussian distribution
    score_low:   Integer   -- The minimum rating
    score_high:  Integer   -- The maximum rating
    out:         String    -- 'matrix' of 'dataframe'

    @Return:
    --------
    df:          DataFrame -- columns = UserId || ItemId || Rating
    OR 
    matrix:      nparray   -- with some nan values depending on density parameter

    @Infos:
    -------
    We consider that each user u has a definite (and random) type t_user(u), from (0, 1, 2, ..., u_unique - 1), 
    that caracterizes the user. Each item i has a definite type t_item(i) too, from (0, 1, ..., i_unique - 1). 
    We then pick a rating r(t_user, t_item) from Unif(score_low, score_high) for all tuples (t_user, t_item). 
    All rating r_hat(i, j) = r_hat(t_user(i), t_item(i)) = r(t_user(i), t_item(i)) + N(0, noise) where N is the 
    Gaussian distribution.
    """
    # Array of user, there are u users, each user has a type from (0, 1, ..., u_unique - 1)
    X = np.random.randint(u_unique, size=u)
    # Array of item, there are i items, each item has a type from (0, 1, ..., i_unique - 1)
    Y = np.random.randint(i_unique, size=i)
    
    # To get the rating between user u (type tu) and item i (type ti), we build a matrix that 
    # associates a random rating between all types tu and ti
    rating_unique_matrix = np.random.randint(low=score_low, high=score_high + 1, size=(u_unique, i_unique))
        
    # We then build the rating matrix
    # Each ratings is r_hat(i,j) = r(i,j) + N(0, noise)
    ratings = np.clip(
                      np.fromfunction(
                                      np.vectorize(lambda i, j: rating_unique_matrix[X[i]][Y[j]] + (np.random.normal(0, noise) if noise > 0 else 0)),
                                      (u, i), 
                                      dtype=int
                                     ), 
                      score_low, 
                      score_high)
    
    # We apply the density parameter
    ratings_nan = np.where(np.random.binomial(2, density, size=(u, i)) == 0, np.nan, 1)*ratings
    
    if out == 'matrix':
        return ratings, ratings_nan
    
    not_nan_index = np.argwhere(~np.isnan(ratings_nan))
    df_nan = pd.DataFrame(not_nan_index)
    df_nan.columns = [USER_ID, ITEM_ID]
    df_nan[RATING] = df_nan.apply(lambda row: ratings_nan[row[USER_ID]][row[ITEM_ID]], axis = 1)

    
    df = pd.DataFrame([[user, item] for user in range(u) for item in range(i)])
    df.columns = [USER_ID, ITEM_ID]
    df[RATING] = df.apply(lambda row: ratings[row[USER_ID]][row[ITEM_ID]], axis = 1)
                              
    return df, df_nan

def fromDFtoDenseMatrix(df):
    """
    @Parameters:
    ------------
    df:    DataFrame -- columns = UserId || ItemId || Rating
    
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