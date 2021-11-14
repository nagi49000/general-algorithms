from pytest import approx, raises
import numpy as np
import numba
from scipy.spatial import distance
from ..src.general_algorithms.true_range_multilateration import TrueRangeMultilateration


def test_error_cases():
    metric_fn = distance.euclidean
    # number of distances not the same as number of vectors
    with raises(ValueError):
        TrueRangeMultilateration(np.zeros((3, 2)), np.zeros(5), metric_fn)

    # number of vectors equal to number of dimensions; just about determined
    TrueRangeMultilateration(np.zeros((5, 5)), np.zeros(5), metric_fn)
    # number of vectors less than dimension; under-determined
    with raises(ValueError):
        TrueRangeMultilateration(np.zeros((5, 6)), np.zeros(5), metric_fn)


def test_random_euclidean():
    np.random.seed(0)
    metric_fn = distance.euclidean
    n_vec = 20
    dim = 5
    anchors = np.random.rand(n_vec, dim)
    answer = np.random.rand(dim)  # we'll solve for this from distances
    distances = np.array([metric_fn(x, answer) for x in anchors])
    trm = TrueRangeMultilateration(anchors, distances, metric_fn)
    soln = trm.solve()
    # check the solution coming out looks like the answer started with
    assert list(answer) == [approx(x) for x in [0.67781654, 0.27000797, 0.73519402, 0.96218855, 0.24875314]]
    assert list(soln.x) == [approx(x) for x in [0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152]]


def test_random_euclidean_with_noise():
    np.random.seed(0)
    metric_fn = distance.euclidean
    n_vec = 20
    dim = 5
    anchors = np.random.rand(n_vec, dim)
    answer = np.random.rand(dim)  # we'll solve for this from distances
    distances = np.array([metric_fn(x, answer) for x in anchors]) + np.random.rand(n_vec)
    trm = TrueRangeMultilateration(anchors, distances, metric_fn)
    soln = trm.solve()
    assert list(answer) == [approx(x) for x in [0.67781654, 0.27000797, 0.73519402, 0.96218855, 0.24875314]]
    assert list(soln.x) == [approx(x) for x in [0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152]]


def test_random_cosine():

    @numba.jit(nopython=True)
    def numba_cosine(x, y):
        eps = 1.0e-16
        return 1.0 - np.dot(x, y) / (eps + np.sqrt(np.dot(x, x) * np.dot(y, y)))

    np.random.seed(0)
    metric_fn = distance.cosine
    n_vec = 20
    dim = 5
    anchors = np.random.rand(n_vec, dim)
    answer = np.random.rand(dim)  # we'll solve for this from distances
    distances = np.array([metric_fn(x, answer) for x in anchors])

    trm = TrueRangeMultilateration(anchors, distances, metric_fn)
    soln = trm.solve()
    assert list(answer) == [approx(x) for x in [0.67781654, 0.27000797, 0.73519402, 0.96218855, 0.24875314]]
    assert list(soln.x) == [approx(x) for x in [0.67818882, 0.43758721, 0.89177300, 0.96366276, 0.38344152]]

    # verify running with a numba fn as the distance function. Since scipy.optimize.minimize calls the residual
    # function so often, using a custom optimized function can give significant speed up
    trm = TrueRangeMultilateration(anchors, distances, numba_cosine)
    soln = trm.solve()
    assert list(answer) == [approx(x) for x in [0.67781654, 0.27000797, 0.73519402, 0.96218855, 0.24875314]]
    assert list(soln.x) == [approx(x) for x in [0.67818882, 0.43758721, 0.89177300, 0.96366276, 0.38344152]]
