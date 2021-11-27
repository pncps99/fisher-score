import numpy as np
from scipy.stats import zscore

def fisher_score(X, y, gamma=0.125):
    """
    Computes the Fisher Score for X.

    Parameters
    ----------
    X: array
        n*d array of training samples
    y: array
        n*1 array of training targets
    gamma: float, optional
        regularization parameter

    Returns
    -------
    F: array
        1*d array of Fisher Score's for each feature
    """

    c = np.unique(y)
    n = X.shape[0]

    def csize(k):
        '''Calculates size of class k'''

        return X[y==k].size

    def _mu(k):
        '''Calculates the mean vector for class k'''

        return np.mean(X[y==k],axis=0)

    mu = np.sum([csize(k)*_mu(k) for k in c], axis=0)

    Sb = np.sum([csize(k)*(_mu(k)-mu).reshape(-1,1)*(_mu(k)-mu) for k in c], axis=0)

    St = np.sum([(zscore(X[i])-mu).reshape(-1,1)*(zscore(X[i])-mu) for i in range(n)], axis=0)
    
    F = np.diagonal(Sb*np.linalg.inv(St+gamma*np.identity(St.shape[0])))

    return F, np.argsort(F,0)[::-1]