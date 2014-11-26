"""

Gaussian process regression in 1D using a fixed kernel.

Author:
    Ilias Bilionis

Date:
    11/25/2014

"""


import GPy
import numpy as np


def _check_and_regularize_1d_array(x):
    """
    Checks if `x` is a `np.ndarray`. If `x.ndim` is 1,
    then, it turns it into a column vector.
    Otherwise, it leaves it as is.
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x[:, None]
    assert x.ndim == 2
    return x


def run(x, y,
        variance=1.,
        length_scale=1.,
        noise_variance=1.,
        optimize=False,
        x_eval=None,
        num_samples=10
        ):
    """
    Perform 1D regression.

    :param x:               The observed inputs (1D `np.ndarray`)
    :param y:               The observed outputs (1D `np.ndarray`).
    :param variance:        The signal strength of the square exponential
                            covariance function (positive float).
    :param length_scale:    The length scale of the square
                            exponential covariance function (positive float).
    :param noise_variance:  The noise of the model (non-negative float).
    :param optimize:        If `True` then the model is optimized
                            by maximizing the marginal likelihood
                            with respect to the hyper-parameters (bool).
    :param x_eval:          The points at which the predictive distribution
                            should be evaluated. If `None`, then it
                            is set to `np.linspace(x.min(), x.max(), 100)`
                            (1D `np.ndarray` or `NoneType`).
    :param num_samples:     Number of samples to take from the predictive
                            distribution (positive integer).
    :returns:               A dictionary containing the following elements:
                                + x_eval:       points on which the predictive
                                                distribution is actually evaluated
                                                (1D `np.ndarray`)
                                + y_mu:         the mean of the predictive distribution
                                                (1D `np.ndarray` of size x_eval.shape[0])
                                + y_var:        the predictive variance
                                                (1D `np.ndarray` of size x_eval.shape[0])
                                + y_q05:        the 5% lower quantile of the predictive
                                                distribution of the GP
                                                (1D `np.ndarray` of size x_eval.shape[0])
                                + y_q95:        the 5% lower quantile of the predictive
                                                distribution of the GP
                                                (1D `np.ndarray` of size x_eval.shape[0])
                                + y_s:          samples from the predictive distribution
                                                of the Gaussian process
                                                (2D `np.ndarray` of size
                                                 num_samples x x_eval.shape[0])
                                + model:        the trained gaussian process model
    """
    # Check input
    x = _check_and_regularize_1d_array(x)
    y = _check_and_regularize_1d_array(y)
    variance = float(variance)
    assert variance > 0.
    length_scale = float(length_scale)
    assert length_scale > 0.
    noise_variance = float(noise_variance)
    assert noise_variance >= 0.
    optimize = bool(optimize)
    if x_eval is None:
        x_eval = np.linspace(x.min(), x.max(), 100)[:, None]
    num_samples = int(num_samples)
    assert num_samples >= 0
    # Initialize the kernel
    k = GPy.kern.RBF(1, lengthscale=length_scale, variance=variance)
    # Initialize the mod
    model = GPy.models.GPRegression(x, y, kernel=k, normalizer=True)
    print '=' * 80
    print '1D Gaussian Process Regression Demo'.center(80)
    print 'Author:'
    print '\tIlias Bilionis (ibilion@purdue.edu)'
    print '\tPredictiveScience Lab'
    print '\tSchool of Mechanical Engineering'
    print '\tPurdue University, West Lafayette, IN, USA'
    print ''
    print 'Powered by:'
    print '\tThe excellent GPy package from the University of Sheffield'
    print '\thttps://github.com/SheffieldML/GPy'
    print '=' * 80

    print 'Initial Model'.center(80)
    print '-' * 80
    print str(model)
    print '-' * 80
    if optimize:
        model.optimize()
        print 'Optimized Model'.center(80)
        print '-' * 80
        print str(model)
        print '-' * 80

if __name__ == '__main__':
    data = np.loadtxt('data/Rat43.dat')
    x = data[:, 0]
    y = data[:, 1]
    out = run(x, y, optimize=True)
