import numpy as np
from scipy.optimize import minimize


class TrueRangeMultilateration:
    """ Non linear least squares solution for over-determined True Range Multilateration
        as described on https://www.cs.ox.ac.uk/files/2663/RR-09-16.pdf
    """

    def __init__(self, anchors, distances, metric_fn):
        """ anchors - numpy.array - 2d array of shape (n_vec, dimension)
            distances - numpy.array - 1d array of shape (n_vec,) of positive numbers
            metric_fn - python function - takes a pair of 1d numpy arrays (of length dimension) and
                                          returns a semi-positive number
        """
        self._anchors = np.array(anchors)
        self._distances = np.array(distances)
        self._metric_fn = metric_fn
        n_vec, dim = self._anchors.shape
        if self._distances.shape != (n_vec,):
            raise ValueError(
                f'odd shapes, anchors.shape = {self._anchors.shape}, distances.shape = {self._distances.shape}'
            )
        if n_vec < dim:
            raise ValueError(f"number of anchors {n_vec} is less than dimesnion {dim}")

    def solve(self, x0=None, method='Nelder-Mead'):
        """ x0 - np.array - initial guess of where the solution is. None finds a 'best guess' starting point
            method - str - method to pass to scipy.optimize.minimize

            returns - scipy OptimizationResult - multilateration solution by minimizing squared errors
        """
        if x0 is None:  # initial guess as closest anchor
            x0 = self._anchors[np.argmin(self._distances)]

        sum_sqr_distances = np.sum(self._distances**2)

        def residual(vec):
            vec_diffs = self._anchors - vec[np.newaxis, :]
            dists = np.array([self._metric_fn(x, x) for x in vec_diffs])
            return np.sum(dists**2) - sum_sqr_distances

        return minimize(residual, x0, method=method)
