"""
This class implements Non-linear Matrix Factorization with Gaussian Processes (NLMFGP), described in:
    Lawrence, Neil D., and Raquel Urtasun.
    "Non-linear matrix factorization with Gaussian processes."
    Proceedings of the 26th Annual International Conference on Machine Learning. ACM, 2009.
"""

import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
import time

# Add parent directory to python path
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from data_fetching.data_set import DataSet


class GpMf():
    def __init__(self, latent_dim, nb_data):
        self.latent_dim = latent_dim
        self.nb_data = nb_data
        self.X = np.random.normal(0, 1e-3, (nb_data, latent_dim))
        self.lin_variance = 1.0
        self.bias_variance = 0.11
        self.white_variance = 5.0
        self.y = None
        self.rated_items = None

    def log_likelihood(self):
        """return the log likelihood of the model"""
        Cj_invy, logDetC = self.invert_covariance()
        yj = np.asmatrix(self.y).T
        Nj = len(self.rated_items)
        likelihood = - 0.5 * (Nj * np.log(2 * math.pi) + logDetC + yj.T.dot(Cj_invy))
        return float(likelihood)

    def invert_covariance(self, gradient=False, nonlinear =False, kernel=linear_kernel):
        q = self.latent_dim
        Nj = len(self.rated_items)
        Xj = np.asmatrix(self.X[self.rated_items, :])
        yj = np.asmatrix(self.y).T
        s_n = self.white_variance
        s_w = self.lin_variance
        s_b = self.bias_variance
        sigNoise = s_w / s_n

        if Nj > q and not nonlinear: # we use the matrix inversion lemma
            XTX = Xj.T * Xj
            B = np.eye(q) + sigNoise * XTX
            Binv = np.linalg.pinv(B)
            _, logdetB = np.linalg.slogdet(B)
            if gradient:
                AinvX = (Xj - sigNoise * Xj * (Binv * XTX)) / s_n
                AinvTr = (Nj - sigNoise * (np.multiply(Xj * Binv, Xj)).sum()) / s_n
            Ainvy = (yj - sigNoise * Xj * (Binv * (Xj.T * yj))) / s_n
            sumAinv = (np.ones((Nj, 1)) - sigNoise * Xj * (Binv * Xj.sum(axis=0).T)) / s_n  # this is Nx1
            sumAinvSum = sumAinv.sum()
            denom = 1 + s_b * sumAinvSum
            fact = s_b / denom
            if gradient:
                CinvX = AinvX - fact * sumAinv * (sumAinv.T * Xj)
                CinvSum = sumAinv - fact * sumAinv * sumAinvSum
                CinvTr = AinvTr - fact * sumAinv.T * sumAinv

            Cinvy = Ainvy - fact * sumAinv * float(sumAinv.T * yj)
            if not gradient:
                logdetA = Nj * np.log(s_n) + logdetB
                logdetC = logdetA + np.log(denom)

        else :
            C = s_w * kernel(Xj, Xj)
            C = C + s_b + s_n * np.eye(Nj)
            Cinv = np.linalg.pinv(C)
            Cinvy = Cinv * yj
            if gradient:
                CinvX = Cinv * Xj
                CinvTr = np.trace(Cinv)
                CinvSum = Cinv.sum(axis=1)
            else:
                _, logdetC = np.linalg.slogdet(C)

        if gradient:
            return Cinvy, CinvSum, CinvX, CinvTr
        else:
            return Cinvy, logdetC

    def log_likelihood_grad(self, ):
        """Computes the gradient of the log likelihood"""
        s_w = self.lin_variance
        s_b = self.bias_variance
        s_n = self.white_variance

        yj = np.asmatrix(self.y).T
        Xj = np.asmatrix(self.X[self.rated_items, :])

        Cinvy, CinvSum, CinvX, CinvTr = self.invert_covariance(gradient=True)
        covGradX = 0.5 * (Cinvy * (Cinvy.T * Xj) - CinvX)
        gX = s_w * 2.0 * covGradX
        gsigma_w = np.multiply(covGradX, Xj).sum()
        CinvySum = Cinvy.sum()
        CinvSumSum = CinvSum.sum()
        gsigma_b = 0.5 * (CinvySum * CinvySum - CinvSumSum)
        gsigma_n = 0.5 * (Cinvy.T * Cinvy - CinvTr)
        return gX, float(gsigma_w), float(gsigma_b), float(gsigma_n)

    def objective(self):
        return -self.log_likelihood()


def fit(dataset, model, nb_iter=10, seed=42, momentum=0.9):
    data = dataset.get_df()
    param_init = np.zeros((1, 3))
    X_init = np.zeros(model.X.shape)
    for iter in range(nb_iter):
        print("iteration", iter)
        tic = time.time()
        np.random.seed(seed=seed)
        state = np.random.get_state()
        users = np.random.permutation(dataset.get_users())
        for user in users:
            #print("begin user", user,  "=========================")
            toc = time.time()
            lr = 1e-4
            y = dataset.get_ratings_user(user)
            rated_items = dataset.get_items_user(user) - 1
            model.y = y
            model.rated_items = rated_items
            grad_X, grad_w, grad_b, grad_n = model.log_likelihood_grad()
            gradient_param = np.array([grad_w * model.lin_variance,
                               grad_b * model.bias_variance,
                               grad_n * model.white_variance])
            param = np.log(np.array([model.lin_variance,
                                     model.bias_variance,
                                     model.white_variance]))
            # update X
            X = X_init[rated_items, :]
            ar = lr * 10
            X = X * momentum + grad_X * ar
            X_init[rated_items, :] = X
            model.X[rated_items, :] = model.X[rated_items, :] + X

            # update variances
            param_init = param_init * momentum + gradient_param * lr
            param = param + param_init
            model.lin_variance = math.exp(param[0, 0])
            model.bias_variance = math.exp(param[0, 1])
            model.white_variance = math.exp(param[0, 2])
            #print("end user", user, "=========================")

        print("end iteration", iter,  "=========================")
        print("duration iteration", time.time() - tic)
    return model


def predict(user, test_items, model, dataset):
    y = dataset.get_ratings_user(user)
    rated_items = dataset.get_items_user(user) - 1
    model.rated_items = rated_items
    model.y = y
    X_test = np.asmatrix(model.X[test_items, :])
    X = np.asmatrix(model.X[model.rated_items, :])
    Cinvy, CinvSum, CinvX, CinvTr = model.invert_covariance(gradient=True)
    mean = model.lin_variance* X_test*(X.T*Cinvy) + Cinvy.sum() * model.bias_variance
    return mean


def perf_weak(dataset, base_dim=11):
    print('Fetch data set...')
    if dataset.dataset == "movielens":
        norm_coeff = 1.6
    else :
        norm_coeff = 6.67
    print('Data set fetched')
    print("Dataset desctiption", dataset.get_description())
    model_init = GpMf(latent_dim=base_dim, nb_data=dataset.item_index_range)
    print('Fit the model...')
    model = fit(dataset=dataset, model=model_init)
    print('Model fitted')
    predictions = []
    true_ratings = []
    test_users = dataset.get_users_test()
    nb_users_test = len(test_users)
    print("nb_users", nb_users_test)
    count = 0
    for user in test_users:
        prediction = predict(user, dataset.get_item_test(user) - 1, model, dataset)
        if prediction > dataset.high_rating:
            prediction = dataset.high_rating
        if prediction < dataset.low_rating:
            prediction = dataset.low_rating
        predictions.append(prediction)
        rating = dataset.get_rating_test(user)
        true_ratings.append(rating)
        count += 1
        print(count, "over ", nb_users_test, "users")
    predictions = np.asarray(predictions)
    true_ratings = np.asarray(true_ratings)
    rmse = np.linalg.norm(predictions - true_ratings) / np.sqrt(nb_users_test)
    nmae = np.sum(np.abs(true_ratings - predictions)) * 1. / (len(predictions) * norm_coeff)
    print("rmse", rmse)
    print("nmae", nmae)
    return float(rmse), float(nmae)

# ========================== DEMO ======================================================================================
def plot_errors_vs_latent_dims():
    base_dims = range(5, 20)
    rmse_res = []
        #0.9192805345815771, 0.9191762806556765, 0.9298582192495648, 0.9264887573748214, 0.942721413670847,
        #0.9464276649204383, 0.9619938797137044, 0.959524838407498, 0.9637647705458857, 0.9654112679112156,
        #0.964014562524729, 0.9828162923399141, 0.9791815989138641, 0.9755912980143179, 0.9909916313775403]
    nmae_res = []
        #0.44971496453721255, 0.44916990674349777, 0.4520802986328138, 0.45121143493891314, 0.45841600020196077,
        #0.4600831341641973, 0.46819195277564857, 0.4653186575931832, 0.4675059307434209, 0.46521752440367997,
        #0.4693989965956004, 0.4749311584164483, 0.47302261002602103, 0.47072878663284334, 0.4775780782441432]

    if not len(rmse_res):
        dataset_movielens_s = DataSet(dataset="movielens", size="S")
        for dim in base_dims:
            (rmse, nmae) = perf_weak(dataset=dataset_movielens_s, base_dim=dim)
            rmse_res.append(rmse)
            nmae_res.append(nmae)

        print(rmse_res)
        print(nmae_res)

    plt.figure()
    plt.plot(base_dims, rmse_res, marker='.')
    plt.grid()
    plt.xlabel('Number of latent dimensions ($r$)')
    plt.ylabel('RMSE (MovieLens 100k)')
    plt.show()

    plt.figure()
    plt.plot(base_dims, nmae_res, marker='*')
    plt.xlabel('Number of latent dimensions ($r$)')
    plt.ylabel('NMAE (MovieLens 100k)')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    print('START')
    # MovieLens dataset 100k
    perf_weak(dataset=DataSet(dataset="movielens", size="S"))
    # MovieLens dataset 1M
    #perf_weak(dataset=DataSet(dataset="movielens", size="M"))
    # Toy dataset
    #perf_weak(dataset=DataSet(dataset="toy"))
    # Jester dataset
    #perf_weak(dataset=DataSet(dataset="jester"))
    # plot_errors_vs_latent_dims()
    print('END')
