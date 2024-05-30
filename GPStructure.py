import numpy as np
import scipy as sp

class GaussianProcess():
    def __init__(self, X1, Y1, kernel_func=None, noise=1e-4):
        # X1: (N x 14) inputs of training data: 
            # states and state differentials = [ds, de_y, e_psi^1, v_x^1, e_y^2, e_psi^2, v_x^2, w^2, k]
        # Y1: (N x 2) outputs of training data, 
            # future state(s) of opponents = [s, e_y, e_psi, v_x, w]
        # kernel_func: (function) a function defining your kernel. 
        # It should take the form f(X1, X2) = K, where K is N x N if X1, X2 are N x k.
        # where k is the number of feature dimensions

        self.noise = noise
        self.X1 = X1
        self.Y1 = Y1
        self.kernel_func = kernel_func
        self.compute_training_covariance()


    def compute_training_covariance(self):
        # Computes the training covariance matrix Σ11(X, X) using self.kernel_func and your input data self.X1
        # Kernel of the observations
        self.Σ11 = self.kernel_func(self.X1, self.X1) + self.noise * np.eye(len(self.X1))


    def compute_posterior(self, X):
        # X: (N x k) set of inputs used to predict
        # μ2: (N x 1) GP means at inputs X
        # Σ2: (N x N) GP means at inputs X

        sigma_inv = np.linalg.inv(self.Σ11)
        sigma_X_X = self.kernel_func(X, X)
        sigma_X1_X = self.kernel_func(self.X1, X)
        sigma_inter = sigma_X1_X.T @ sigma_inv

        # Compute posterior mean
        μ2 = sigma_inter @ self.Y1

        # Compute the posterior covariance
        Σ2 = sigma_X_X - sigma_inter @ sigma_X1_X

        return μ2, Σ2  # posterior mean, covariance
    



###### KERNELS ######

"""Radial basis kernel"""
def radial_basis(X1, X2, sig=1.0, l=0.5):
    N1, k1 = np.shape(X1)
    N2, k2 = np.shape(X2)
    X1 = np.reshape(X1, (N1, 1, k1))
    X2 = np.reshape(X2, (1, N2, k2))
    diff_norm = np.linalg.norm(X1 - X2, axis=-1) ** 2
    K = sig**2 * np.exp(-1/(2*l**2) * diff_norm)
    return K


"""Matern kernel"""
def matern(X1, X2, l=0.5):
    N1, k1 = np.shape(X1)
    N2, k2 = np.shape(X2)
    X1 = np.reshape(X1, (N1, 1, k1))
    X2 = np.reshape(X2, (1, N2, k2))
    diff_norm = np.linalg.norm(X1 - X2, axis=-1) ** 2
    holder = (np.sqrt(3)/l)*diff_norm
    K = (1 + holder) * np.exp(-holder)
    return K