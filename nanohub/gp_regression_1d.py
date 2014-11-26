"""

Gaussian process regression in 1D using a fixed kernel.

Author:
    Ilias Bilionis

Date:
    11/25/2014

"""


import GPy
import numpy as np
from optparse import OptionParser
import cPickle as pickle


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
                                + y_025:        the 2.5% lower quantile of the predictive
                                                distribution of the GP
                                                (1D `np.ndarray` of size x_eval.shape[0])
                                + y_975:        the 97.5% lower quantile of the predictive
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
    model = GPy.models.GPRegression(x, y, kernel=k,
                                    normalizer=True)
    model.Gaussian_noise.variance = noise_variance
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
        print 'Optimizing'.center(80)
        print '-' * 80
        model.optimize(optimizer='lbfgs', messages=True)
        print '-' * 80
        print 'Optimized Model'.center(80)
        print '-' * 80
        print str(model)
        print '-' * 80
    y_mu, y_var = model.predict(x_eval)
    y_025, y_975 = model.predict_quantiles(x_eval)
    return locals()




if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-i', '--input', dest='input_file',
                      default='test.dat',
                      help='the input file (TXT with two columns)')
    parser.add_option('-l', '--length-scale', dest='length_scale',
                      default=1.,
                      help='the length scale of the SE kernel')
    parser.add_option('-s', '--signal-strength', dest='signal_strength',
                      default=1.,
                      help='the signal strength of the SE kernel')
    parser.add_option('-n', '--noise-variance', dest='noise_variance',
                      default=0.1,
                      help='the noise variance added to the SE kernel')
    parser.add_option('--optimize', action='store_true', dest='optimize',
                      default=False,
                      help='optimize the marginal likelihood or not')
    parser.add_option('-o', '--output', dest='output_file',
                      default=None,
                      help='specify the output file')
    parser.add_option('--plot', action='store_true', dest='plot',
                      default=False,
                      help='plot the results')
    parser.add_option('--fig-file', dest='fig_file',
                      default=None,
                      help='specify the png file')
    options, args = parser.parse_args()
    data = np.loadtxt(options.input_file)
    x = data[:, 0]
    y = data[:, 1]
    out = run(x, y, optimize=options.optimize,
              length_scale=options.length_scale,
              variance=options.signal_strength,
              noise_variance=options.noise_variance)
    if options.output_file is None:
        options.output_file = options.input_file + '.out'
    out_data = np.hstack([out['x_eval'], out['y_mu'], out['y_025'], out['y_975']])
    np.savetxt(options.output_file, out_data)
    trained_model_file = options.input_file + '.pcl'
    with open(trained_model_file, 'wb') as fd:
        pickle.dump(out['model'], fd, protocol=pickle.HIGHEST_PROTOCOL)
    if options.plot:
        import matplotlib.pyplot as plt
        plt.plot(x, y, 'r+', markersize=10, markeredgewidth=2)
        plt.plot(out['x_eval'], out['y_mu'], 'b', linewidth=2, label='mean')
        plt.fill_between(out['x_eval'].flatten(), out['y_025'].flatten(),
                         out['y_975'].flatten(), color='grey', alpha=0.5,
                         label='predictive interval')
        if options.fig_file is None:
            options.fig_file = options.input_file + '.png'
        plt.savefig(options.fig_file)
