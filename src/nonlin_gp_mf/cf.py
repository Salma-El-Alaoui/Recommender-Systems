import sys
import os
import numpy as np
import math

# Add parent directory to python path
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from data_fetching.data_fetching import USER_ID, ITEM_ID, RATING, getDataframe, fromDFtoDenseMatrix, getDataframe_toy

def get_data():
    data = getDataframe()
    item_ids = data[ITEM_ID]
    print(item_ids.apply(lambda x: x==242).sum())
    user_ids = data[USER_ID]

class gp_mf():
    def __init__(self, latent_dim, nb_data):
        self.latent_dim = latent_dim
        self.nb_data = nb_data
        self.X = np.random.normal(0, 1e-6, (nb_data, latent_dim))
        self.lin_variance = 1.0
        self.bias_variance = 0.11
        self.white_variance = 5.0
        self.y = None
        self.rated_items = None

    def log_likelihood(self):
        """return the log likelihood of the model"""
        Cj_invy, logDetC = self.invert_covariance()
        yj = np.asmatrix(self.y)
        Nj = len(self.rated_items)
        likelihood = - 0.5 * (Nj * np.log(2 * math.pi) + logDetC + yj.T.dot(Cj_invy))
        return likelihood

    def invert_covariance(self, gradient=False):
        q = self.latent_dim
        Nj = len(self.rated_items)
        Xj = np.asmatrix(self.X[self.rated_items, :])
        yj = np.asmatrix(self.y).T
        s_n = self.white_variance
        s_w = self.lin_variance
        s_b = self.bias_variance
        sigNoise = s_w / s_n

        if Nj > q: # we use the matrix inversion lemma
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
            C = s_w * Xj * Xj.T
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

def test():
    user = 3
    # shape = (#items, #users)
    ratings_matrix = fromDFtoDenseMatrix(getDataframe_toy()).T
    model = gp_mf(latent_dim=250, nb_data=ratings_matrix.shape[0])
    # vector of observed ratings by this user
    all_y = ratings_matrix[:, user]
    model.y = all_y[~np.isnan(all_y)]
    # indices of items rated by this user
    model.rated_items = np.where(~np.isnan(all_y))
    print("cov", model.invert_covariance()[0].shape)

test()


