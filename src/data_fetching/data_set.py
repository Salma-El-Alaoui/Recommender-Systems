import io
import csv
import pandas as pd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from functools import lru_cache
from pathlib import Path



class DataSet:

    def __init__(self, dataset='movielens', size='S'):
        """
           @Parameters:
           ------------
        dataset: String - - 'movielens' or 'jester'
        size:    String - - 'S', 'M' or 'L'(only for 'movielens')

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
        self.USER_ID = 'user_id'
        self.ITEM_ID = 'item_id'
        self.RATING = 'rating'
        self.TIMESTAMP = 'timestamp'

        # Define some constants
        self.dataset = dataset
        self.size = size
        self.DATASETS = ['movielens', 'jester']
        self.DATASET_WITH_SIZE = ['movielens']
        self.SIZE = ['S', 'M', 'L']


        # Check inputs
        if self.dataset not in self.DATASETS:
            raise NameError("This dataset is not allowed.")
        if self.size not in self.SIZE and dataset in self.DATASET_WITH_SIZE:
            raise NameError("This size is not allowed.")

        # Change size if necessary
        self.size = size if dataset in self.DATASET_WITH_SIZE else 'unique'

        # Configure url, filename, separator and columns in csv
        if self.dataset == 'movielens':
            self.url_map = {
                'S': "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
                'M': "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
                'L': "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
            }

            self.filename_map = {
                'S': "ml-100k/u.data",
                'M': "ml-1m/ratings.dat",
                'L': "ml-10m/ratings.dat"
            }

            self.separator_map = {
                'S': '\t',
                'M': '::',
                'L': '::'
            }

            self.columns_map = {
                'S': [self.USER_ID, self.ITEM_ID, self.RATING, self.TIMESTAMP],
                'M': [self.USER_ID, self.ITEM_ID, self.RATING, self.TIMESTAMP],
                'L': [self.USER_ID, self.ITEM_ID, self.RATING, self.TIMESTAMP]
            }

        if dataset == 'jester':
            self.url_map = {
                'unique': "http://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip",
            }

            self.filename_map = {
                'unique': "jester_ratings.dat",
            }

            self.separator_map = {
                'unique': '\t\t'
            }

            self.columns_map = {
                'unique': [self.USER_ID, self.ITEM_ID, self.RATING],
            }

        self.df = self.get_df()
        self.nb_users = len(np.unique(self.df[self.USER_ID]))
        self.nb_items = len(np.unique(self.df[self.ITEM_ID]))
        self.low_user = np.min(self.df[self.USER_ID])
        self.high_user = np.max(self.df[self.USER_ID])

    @lru_cache(maxsize=256)
    def get_df(self):
        """
        @Return:
        --------
        df:      DataFrame -- columns = UserId || ItemId || Rating

        """
        # Load data in memory
        csv_ondisk = Path("../../csv/" + self.filename_map[self.size])
        if csv_ondisk.is_file():
            df = pd.read_csv(csv_ondisk, sep=self.separator_map[self.size], header=None)
        else:
            url = urlopen(self.url_map[self.size])
            zipfile = ZipFile(BytesIO(url.read()))
            unzipfile = io.TextIOWrapper(zipfile.open(self.filename_map[self.size], 'r'))
            df = pd.read_csv(unzipfile, sep=self.separator_map[self.size], header=None)
        df.columns = self.columns_map[self.size]
        df = df[[self.USER_ID, self.ITEM_ID, self.RATING]]

        return df

    def get_df_toy(self, u=100, i=1000, u_unique=10, i_unique=5, density=0.1, noise=0.3, score_low=0, score_high=5,
                   out='dataframe'):
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
        ratings = np.round(
                    np.clip(
                      np.fromfunction(
                                      np.vectorize(lambda i, j: rating_unique_matrix[X[i]][Y[j]] + (np.random.normal(0, noise) if noise > 0 else 0)),
                                      (u, i),
                                      dtype=int
                                     ),
                          score_low,
                          score_high
                           ),
                         2)

        # We apply the density parameter
        ratings_nan = np.where(np.random.binomial(2, density, size=(u, i)) == 0, np.nan, 1)*ratings

        if out == 'matrix':
            return ratings_nan

        not_nan_index = np.argwhere(~np.isnan(ratings_nan))
        df_nan = pd.DataFrame(not_nan_index)
        df_nan.columns = [self.USER_ID, self.ITEM_ID]
        df_nan[self.RATING] = df_nan.apply(lambda row: ratings_nan[row[self.USER_ID]][row[self.ITEM_ID]], axis=1)

        df = pd.DataFrame([[user, item] for user in range(u) for item in range(i)])
        df.columns = [self.USER_ID, self.ITEM_ID]
        df[self.RATING] = df.apply(lambda row: ratings[row[self.USER_ID]][row[self.ITEM_ID]], axis=1)
                              
        return df, df_nan

    def split_train_test(self, nb_users_weak=5000, per_users_weak=None):
        # not complete
        if per_users_weak:
            nb_users_weak = int(self.nb_users * per_users_weak)
        ind_users_remove = list(np.random.randint(low=self.low_user, high=self.high_user, size=nb_users_weak))
        users_items_remove = dict.fromkeys(ind_users_remove)
        for key in users_items_remove.keys():
            items_user = list(self.df[self.df.USER_ID == key][self.ITEM_ID])
            users_items_remove[key] = items_user[0]
        df_train = self.df
        df_train[self.RATING] = df_train.apply(
            lambda row: np.nan if (row[self.USER_ID] in ind_users_remove
                                   and row[self.ITEM_ID] == users_items_remove[row[self.USER_ID]])
            else row[self. RATING])

    def df_to_matrix(self):
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
        user_id_max = np.max(self.df.user_id)
        item_id_max = np.max(self.df.item_id)
        res = np.nan*np.zeros((user_id_max + 1, item_id_max + 1))
        for row in self.df.values:
            res[row[0]][row[1]] = row[2]
        return res

    @lru_cache(maxsize=256)
    def get_df_cv(self):
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
                self.df = pd.read_csv(unzipfile, sep='\t', header=None)
                self.df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
                train_test[ext] = self.df
            folds.append(train_test)
        return folds

