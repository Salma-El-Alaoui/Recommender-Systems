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

    #############
    # INTERFACE #
    #############

    """
    // How to use //

    Initialize the class with a dataset ('movielens', 'jester' or 'toy'), e.g:
    ds = DataSet(dataset='movielens')

    Once loaded, get your dataframe, columns = [ user_id, item_id, rating ]:
    df = ds.get_df()

    If the toy dataset was chosen, one can access the full dataset:
    df_complete = ds.get_df_complete()

    Instead of the dataframe, one can get the dense rating matrix:
    dense_matrix = DataSet.df_to_matrix(df)

    To get a train / test dataframe:
    train, test = ds.split_train_test()

    To be continued...
    """

    ####################
    # Static variables #
    ####################

    ## Column names
    USER_ID = 'user_id'
    ITEM_ID = 'item_id'
    RATING = 'rating'
    TIMESTAMP = 'timestamp'

    ## Dataset constants
    DATASETS = ['movielens', 'jester', 'toy'] ## All datasets
    DATASETS_WITH_SIZE = ['movielens']
    DATASETS_TO_BE_FETCHED = ['movielens', 'jester']
    SIZE = ['S', 'M', 'L']

    ###############
    # Constructor #
    ###############

    def __init__(self, dataset='movielens', size='S', out='dataframe',
                     u=100, i=1000, u_unique=10, i_unique=5, density=0.1, noise=0.3, score_low=0, score_high=5):
        """
        @Parameters:
        ------------
        dataset: String -- 'movielens' or 'jester' or 'toy'
        size:    String -- 'S', 'M' or 'L'(only for 'movielens')
        out:     String -- 'dataframe' or 'matrix'
        u, i, u_unique, i_unique, density, noise, score_low, score_high -- See get_df_toy (only for toy dataset)

        @Infos:
        -------
        For movielens:
            -> Size = S:   100K ratings,  1K users, 1.7K movies, ~   2MB, scale: [ 1  , 5 ], density:
            -> Size = M:     1M ratings,  6K users,   4K movies, ~  25MB, scale: [ 1  , 5 ], density: 4.26%
            -> Size = L:    10M ratings, 72K users,  10K movies, ~ 265MB, scale: [0.5 , 5 ], density: 0.52%

            All users have rated at least 20 movies no matter the size of the dataset

        For jester:
            -> Uniq. size: 1.7M ratings, 60K users,  150  jokes, ~  33MB, scale: [-10 , 10], density: 31.5%
               Values are continuous.
        """

        self.dataset = dataset

        # Check inputs
        if dataset not in DataSet.DATASETS:
            raise NameError("This dataset is not allowed.")
        if size not in DataSet.SIZE and dataset in DataSet.DATASETS_WITH_SIZE:
            raise NameError("This size is not allowed.")

        # Configure parameters
        if dataset in DataSet.DATASETS_TO_BE_FETCHED:
            self.__set_params_online_ds(dataset)
        else:
            self.__set_params_toy_ds(u, i, u_unique, i_unique, density, noise, score_low, score_high)

        self.df, self.df_complete = self.__set_df(out)
        self.nb_users  = len(np.unique(self.df[self.USER_ID]))
        self.nb_items  = len(np.unique(self.df[self.ITEM_ID]))
        self.low_user  = np.min(self.df[self.USER_ID])
        self.high_user = np.max(self.df[self.USER_ID])


    ##################
    # Public methods #
    ##################

    def get_df(self):
        return self.df

    def get_df_complete(self):
        # Only for toy dataset
        return self.df_complete

    def split_train_test(self, nb_users_weak=5000, per_users_weak=None):
        # To be complete
        if per_users_weak:
            nb_users_weak = int(self.nb_users * per_users_weak)
        ind_users_remove = list(np.random.randint(low=self.low_user, high=self.high_user, size=nb_users_weak))
        users_items_remove = dict.fromkeys(ind_users_remove)
        for key in users_items_remove.keys():
            items_user = list(self.df[self.df[DataSet.USER_ID] == key][DataSet.USER_ID])
            users_items_remove[key] = items_user[0]
        df_train = self.df
        df_train[self.RATING] = df_train.apply(
            lambda row: np.nan if (row[DataSet.USER_ID] in ind_users_remove
                                   and row[DataSet.ITEM_ID] == users_items_remove[row[DataSet.USER_ID]])
            else row[self.RATING])

    def get_description(self):
        pass

    @staticmethod
    def df_to_matrix(df):
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
        user_id_max = np.max(df[DataSet.USER_ID])
        item_id_max = np.max(df[DataSet.ITEM_ID])
        res = np.nan * np.zeros((user_id_max + 1, item_id_max + 1))
        for row in df.values:
            res[row[0]][row[1]] = row[2]
        return res

    ###################
    # Private methods #
    ###################

    def __set_params_online_ds(self, name, size):
        # Configure url, filename, separator and columns in csv

        # Change size if necessary
        self.__size = size if dataset in DataSet.DATASETS_WITH_SIZE else 'unique'

        if self.dataset == 'movielens':
            self.__url_map = {
                'S': "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
                'M': "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
                'L': "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
            }

            self.__filename_map = {
                'S': "ml-100k/u.data",
                'M': "ml-1m/ratings.dat",
                'L': "ml-10m/ratings.dat"
            }

            self.__separator_map = {
                'S': '\t',
                'M': '::',
                'L': '::'
            }

            self.__columns_map = {
                'S': [DataSet.USER_ID, DataSet.ITEM_ID, DataSet.RATING, DataSet.TIMESTAMP],
                'M': [DataSet.USER_ID, DataSet.ITEM_ID, DataSet.RATING, DataSet.TIMESTAMP],
                'L': [DataSet.USER_ID, DataSet.ITEM_ID, DataSet.RATING, DataSet.TIMESTAMP]
            }

        if self.dataset == 'jester':
            self.__url_map = {
                'unique': "http://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip",
            }

            self.__filename_map = {
                'unique': "jester_ratings.dat",
            }

            self.__separator_map = {
                'unique': '\t\t'
            }

            self.__columns_map = {
                'unique': [DataSet.USER_ID, DataSet.ITEM_ID, DataSet.RATING],
            }

    def __set_params_toy_ds(u, i, u_unique, i_unique, density, noise, score_low, score_high):
        self.__u = u
        self.__i = i
        self.__u_unique = u_unique
        self.__i_unique = i_unique
        self.density = density
        self.__noise = noise
        self.score_low = score_low
        self.score_high = score_high

    @lru_cache(maxsize=256)
    def __set_df(self, out):
        """
        @Return:
        --------
        df:      DataFrame -- columns = UserId || ItemId || Rating

        """
        # Load data in memory
        if self.dataset in DataSet.DATASETS_TO_BE_FETCHED:
            csv_ondisk = Path("../../csv/" + self.__filename_map[self.__size])
            if csv_ondisk.is_file():
                df = pd.read_csv(csv_ondisk, sep=self.__separator_map[self.__size], header=None)
            else:
                url = urlopen(self.__url_map[self.__size])
                zipfile = ZipFile(BytesIO(url.read()))
                unzipfile = io.TextIOWrapper(zipfile.open(self.__filename_map[self.__size], 'r'))
                df = pd.read_csv(unzipfile, sep=self.__separator_map[self.__size], header=None)
            df.columns = self.__columns_map[self.__size]
            df = df[[DataSet.USER_ID, DataSet.ITEM_ID, DataSet.RATING]]
            if out == 'matrix':
                df = self.df_to_matrix(df)
            df_complete = None
        else:
            df, df_complete = self.__get_df_toy(self.__u, self.__i, self.__u_unique, self.__i_unique,
                                       self.density, self.__noise, self.score_low, self.score_high, out)


        return df, df_complete

    def __get_df_toy(self, u, i, u_unique, i_unique, density, noise, score_low, score_high, out):
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
            return ratings, ratings_nan

        not_nan_index = np.argwhere(~np.isnan(ratings_nan))
        df_nan = pd.DataFrame(not_nan_index)
        df_nan.columns = [DataSet.USER_ID, DataSet.ITEM_ID]
        df_nan[DataSet.RATING] = df_nan.apply(lambda row: ratings_nan[row[DataSet.USER_ID]][row[DataSet.ITEM_ID]], axis=1)

        df = pd.DataFrame([[user, item] for user in range(u) for item in range(i)])
        df.columns = [DataSet.USER_ID, DataSet.ITEM_ID]
        df[DataSet.RATING] = df.apply(lambda row: ratings[row[DataSet.USER_ID]][row[DataSet.ITEM_ID]], axis=1)

        return df, df_nan
