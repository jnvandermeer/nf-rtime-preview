from __future__ import division

import unittest

import numpy as np

from wyrm.processing import calculate_whitening_matrix, apply_spatial_filter
from wyrm.types import Data


SAMPLES = 1000
CHANS = 3


class TestCalculateWhitentingMatrix(unittest.TestCase):

    def setUp(self):
        data = np.random.randn(SAMPLES, CHANS)
        data[:, 1] += 0.5 * data[:, 0]
        data[:, 2] -= 0.5 * data[:, 0]
        t = np.arange(SAMPLES)
        chans = ['chan{i}'.format(i=i) for i in range(CHANS)]
        self.cnt = Data(data, [t, chans], ['time', 'channels'], ['ms', '#'])

    def test_shape(self):
        """The whitening filter should have the shape: CHANSxCHANS."""
        a = calculate_whitening_matrix(self.cnt)
        self.assertEqual(a.shape, (CHANS, CHANS))

    def test_diagonal(self):
        """The whitened data should have all 1s on the covariance matrix."""
        a = calculate_whitening_matrix(self.cnt)
        dat2 = apply_spatial_filter(self.cnt, a)
        vals = np.diag(np.cov(dat2.data.T))
        np.testing.assert_array_almost_equal(vals, [1. for i in range(len(vals))])

    def test_zeros(self):
        """The whitened data should have all 0s on the non-diagonals of the covariance matrix."""
        a = calculate_whitening_matrix(self.cnt)
        dat2 = apply_spatial_filter(self.cnt, a)
        cov = np.cov(dat2.data.T)
        # substract the diagonals
        cov -= np.diag(np.diag(cov))
        self.assertAlmostEqual(np.sum(cov), 0)

    def test_calculate_whitening_matrix_copy(self):
        """calculate_whitening_matrix must not modify arguments."""
        cpy = self.cnt.copy()
        calculate_whitening_matrix(self.cnt)
        self.assertEqual(self.cnt, cpy)


if __name__ == '__main__':
    unittest.main()

