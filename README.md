# Recommender-Systems

 --------------------------------------------------
 Collaborative Filtering using Matrix Factorization
 --------------------------------------------------


AUTHORS
-------

Salma El Alaoui,
Olivier Chancé,
Sophia Lazraq,
Liu Zhengying,

STRUCTURES
----------

- README.md 
- src/ 
 	- ALS_WR/
 		- als_wr.py
 	- nonlin_gp_mf/
 		- cf.py
 		- __init__.py
 	- pmf/
 		- pmf.py
 		- __init__.py
 	- data_fetching/
 		- data_set.py
 		- __init__.py

GENERAL INFORMATION
-------------------

This work implements different matrix factorization techniques in the context 
of collaborative filtering. 

- pmf.py     implements Probabilistic Matrix factorization with fixed priors
- cf.py      implements Non-linear Matrix Factorization with Gaussian Processes (NLMFGP)
- als_wr.py  implements Alternating-Least-Squares with Weighted-λ-Regularization

In order to assess our implementation, we use 3 datasets:
- MovieLens  each row corresponds to a rating given by a user to a movie
- Jester     each row corresponds to a rating given by a user to a joke
- ToyDataset each row corresponds to a rating given by a user to an item

All informations about these datasets and techniques are available in the report.

HOW TO USE
----------

### data_set.py
data_set.py is responsible for fetching/creating the desired dataset and spliting it into
training and test sets.
For this part, nothing has to be run in the terminal, but as a quick introduction, main command
are presented:

  Initialize the class with a dataset ('movielens', 'jester' or 'toy'), e.g:
  ds = DataSet(dataset='movielens')

  Once loaded, to get the dataframe with columns = [ user_id, item_id, rating ]:
  df = ds.get_df()

  To get some infos on the df, run:
  ds.get_description()

  To get a train / test dataframe:
  train_df, test_df = ds.get_df_train(), ds.get_df_test()

### pmf.py
pmf.py implements Probabilistic Matrix factorization with fixed priors.

In src/pmf/, run in the terminal:
> python pmf.py

It will compute the RMSE by epoch as well as the excecution and epoch times
for dataset='movielens', size='S'

### als_wr.py
als_wr.py implements Alternating-Least-Squares with Weighted-λ-Regularization.

In src/ALS_WR/, run in the terminal:
> python als_wr.py

It will compute the RMSE by epoch as well as the excecution and epoch times
for dataset='movielens', size='S'

### cf.py.py
cf.py implements Non-linear Matrix Factorization with Gaussian Processes (NLMFGP).

In src/nonlin_gp_mf/, run in the terminal:
> python pmf.py

The dataset used for this one is movielens too, with size='S'

For each method, only a few dataset/options are tested in order to reduce the calculation time
but feel free to explore other options by uncomment line of codes in the main method of
each class.

