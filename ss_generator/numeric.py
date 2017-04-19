import numpy as np

def multivariate_gaussian(x, mean, cov):
    '''Calculate the probability density of a multivariate
    gaussian distribution at x.
    '''
    n = x.shape[0]

    return 1 / (np.power(2 * np.pi, n / 2.0) * np.absolute(np.power(np.linalg.det(cov), 0.5))) \
            * np.exp(-0.5 * np.dot(x - mean, np.dot(np.linalg.inv(cov), x - mean)))

def gaussian(x, mean, std):
    '''Calculate the probability density of a gaussian distribution
    at x.
    '''
    return multivariate_gaussian(np.array([x]), np.array([mean]), np.array([[std]]))
