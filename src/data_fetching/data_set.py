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

    Once loaded, to get the dataframe with columns = [ user_id, item_id, rating ]:
    df = ds.get_df()

    If the toy dataset was chosen, one can access the full dataset:
    df_complete = ds.get_df_complete()

    Instead of the dataframe, one can get the dense rating matrix:
    dense_matrix = DataSet.df_to_matrix(df)
    
    To get some infos on the df, run:
    ds.get_description()

    To get a train / test dataframe:
    train_df, test_df = ds.split_train_test(False)

    Once the model trained, U and V built, one can get the prediction dataframe:
    pred_df = DataSet.U_V_to_df(U, V, None, train_df)

    Finally, to assess the accuracy of the model:
    score = DataSet.get_score(train_df, pred_df)
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

    def __init__(self, dataset='movielens', size='S',
                     u=100, i=1000, u_unique=10, i_unique=5, density=0.1, noise=0.3, score_low=0, score_high=5):
        """
        @Parameters:
        ------------
        dataset: String -- 'movielens' or 'jester' or 'toy'
        size:    String -- 'S', 'M' or 'L'(only for 'movielens')
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
            self.__set_params_online_ds(dataset, size)
        else:
            self.__set_params_toy_ds(u, i, u_unique, i_unique, density, noise, score_low, score_high)

        self.df, self.df_complete = self.__set_df()
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

    def split_train_test(self, strong_generalization = True, train_size = 0.8):
        """
        @Parameters:
        ------------
        strong_generalization: Boolean          -- If false, weak generalization approach
        train_size:            Float in [0, 1]  -- Only for strong_generalization

        @Return:
        --------
        train_set_df, test_set_observed_df, test_set_heldout_df -- if strong generalization approach
        train_set_df, test_set_df                               -- if weak generalization approach

        @Infos:
        -------
        In a nutshell:
        Weak generalization --> For each user, one rating is held out (test set), the other ratings = training set
        Strong generalization --> User set is divided in training set / test set. The model is trained using all 
                                  data available in training set. Test set is then divided in observed values/held out
                                  values. Predictions have to be made on the test set on held out values, based on 
                                  observed values using the model trained on the training set.
        For more informations : https://people.cs.umass.edu/~marlin/research/thesis/cfmlp.pdf - Section 3.3
        """
        unique_user_id = np.unique(self.df[DataSet.USER_ID])
        if strong_generalization:
            user_id_train_set = np.random.choice(unique_user_id, size=int(train_size*len(unique_user_id)), replace=False)
            user_id_test_set  = np.setdiff1d(unique_user_id, user_id_train_set)
            train_set_df = self.df[self.df[DataSet.USER_ID].isin(user_id_train_set)]
            test_set_df = self.df[~self.df[DataSet.USER_ID].isin(user_id_train_set)]
            idx_heldout_test_set = [np.random.choice(test_set_df[test_set_df[DataSet.USER_ID] == idx].index) for idx in user_id_test_set]
            test_set_heldout_df  = test_set_df.loc[idx_heldout_test_set]
            test_set_observed_df = test_set_df.loc[np.setdiff1d(test_set_df.index, idx_heldout_test_set)]
            return train_set_df, test_set_observed_df, test_set_heldout_df
        else:
            # Weak generalization
            idx_heldout_test_set = [np.random.choice(self.df[self.df[DataSet.USER_ID] == idx].index) for idx in unique_user_id]
            test_set_df  = self.df.loc[idx_heldout_test_set]
            train_set_df = self.df.loc[np.setdiff1d(self.df.index, idx_heldout_test_set)]
            return train_set_df, test_set_df

    def get_description(self):
        return {
            "Number of users": self.nb_users,
            "Number of items": self.nb_items,
            "Lowest user": self.low_user,
            "Highest user": self.high_user,
            "Density": self.df.shape[0] / (self.nb_items * self.nb_users),
            "Mean of ratings": np.mean(self.df[DataSet.RATING]),
            "Standard deviation of ratings": np.std(self.df[DataSet.RATING])
        }

    @staticmethod
    def U_V_to_df(U, V, list_index, train_df = None):
        """
        @Parameters:
        ------------
        U:          nparray   -- shape = (#users, k)
        V:          nparray   -- shape = (#items, k)
        traindf:    dataframe -- columns = UserId || ItemId || Rating
        list_index: list      -- shape = [ [user_id_1, item_id_1], [user_id_2, item_id_2], ... ]

        @Return:
        --------
        R_hat:      Dataframe -- columns = UserId || ItemId || Rating

        @Infos:
        -------
        This function is aimed to return all ratings for a list of tuples (user_id, item_id)
        If such a list is not provided, it is built using train_df.
        """

        R_hat = []

        if not (list_index or train_df):
            raise ValueError('Either list_index or train_df has to be provided')

        if train_df:
            list_index = []
            for row in train_df.values:
                list_index.append([row[0], row[1]])

        for index in list_index:
            idx_user = index[0]
            idx_item = index[1]
            R_hat.append([ idx_user, idx_item, np.dot(U[idx_user,:], V[idx_item,:]) ])

        return pd.DataFrame(R_hat)

    @staticmethod
    def get_score(train_df, prediction_df):
        pred_map = {}
        for row in prediction_df.values:
            pred_map[str(row[0]) + '-' + str(row[1])] = row[2]

        score = 0
        for row in train_df.values:
            score += (row[2] - pred_map[str(row[0]) + '-' + str(row[1])])**2

        score /= train_df.shape[0]

        return score

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
        self.__size = size if self.dataset in DataSet.DATASETS_WITH_SIZE else 'unique'

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

    def __set_params_toy_ds(self, u, i, u_unique, i_unique, density, noise, score_low, score_high):
        self.__u = u
        self.__i = i
        self.__u_unique = u_unique
        self.__i_unique = i_unique
        self.density = density
        self.__noise = noise
        self.score_low = score_low
        self.score_high = score_high

    @lru_cache(maxsize=256)
    def __set_df(self):
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
            df_complete = None
        else:
            df_complete, df = self.__get_df_toy(self.__u, self.__i, self.__u_unique, self.__i_unique,
                                       self.density, self.__noise, self.score_low, self.score_high, out="dataframe")


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