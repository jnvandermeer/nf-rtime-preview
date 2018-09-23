from __future__ import division

import unittest

import numpy as np

from wyrm.processing import apply_spatial_filter
from wyrm.processing import swapaxes
from wyrm.types import Data


SAMPLES = 1000
CHANS = 3
EPOS = 5


class TestApplySpatialFilter(unittest.TestCase):

    def setUp(self):
        data = np.random.randn(SAMPLES, CHANS)
        data[:, 1] += 0.5 * data[:, 0]
        data[:, 2] -= 0.5 * data[:, 0]
        t = np.arange(SAMPLES)
        chans = ['chan{i}'.format(i=i) for i in range(CHANS)]
        self.cnt = Data(data, [t, chans], ['time', 'channels'], ['ms', '#'])

        # construct epo
        epo_dat = np.array([data for i in range(EPOS)])
        classes = ['class{i}'.format(i=i) for i in range(EPOS)]
        self.epo = Data(epo_dat, [classes, t, chans], ['class', 'time', 'channels'], ['#', 'ms', '#'])

        # my little spatial filter
        self.w = np.array([[ 0, 0.5, 1],
                           [-1, 0.5, 0],
                           [ 1, 0.5, 0]])


    def test_shape(self):
        """The spatial filtered data should keep its shape."""
        cnt_f = apply_spatial_filter(self.cnt, self.w)
        epo_f = apply_spatial_filter(self.epo, self.w)
        self.assertEqual(self.cnt.data.shape, cnt_f.data.shape)
        self.assertEqual(self.epo.data.shape, epo_f.data.shape)


    def test_spatial_filter_cnt(self):
        """Spatial filtering should work with cnt."""
        cnt_f = apply_spatial_filter(self.cnt, self.w)
        # chan 0
        np.testing.assert_array_equal(cnt_f.data[:, 0], self.cnt.data[:, 2] - self.cnt.data[:, 1])
        # chan 1
        np.testing.assert_array_equal(cnt_f.data[:, 1], 0.5 * np.sum(self.cnt.data, axis=-1))
        # chan 2
        np.testing.assert_array_equal(cnt_f.data[:, 2], self.cnt.data[:, 0])


    def test_spatial_filter_epo(self):
        """Spatial filtering should work with epo."""
        cnt_f = apply_spatial_filter(self.cnt, self.w)
        epo_f = apply_spatial_filter(self.epo, self.w)
        for i in range(EPOS):
            np.testing.assert_array_equal(epo_f.data[i, ...], cnt_f.data)


    def test_prefix_and_postfix(self):
        """Prefix and Postfix are mutual exclusive."""
        with self.assertRaises(ValueError):
            apply_spatial_filter(self.cnt, self.w, prefix='foo', postfix='bar')


    def test_prefix(self):
        """Apply prefix correctly."""
        cnt_f = apply_spatial_filter(self.cnt, self.w, prefix='foo')
        self.assertEqual(cnt_f.axes[-1], ['foo'+str(i) for i in range(CHANS)])


    def test_prefix_w_wrong_type(self):
        """Raise TypeError if prefix is neither None or str."""
        with self.assertRaises(TypeError):
            apply_spatial_filter(self.cnt, self.w, prefix=1)


    def test_postfix(self):
        """Apply postfix correctly."""
        cnt_f = apply_spatial_filter(self.cnt, self.w, postfix='foo')
        self.assertEqual(cnt_f.axes[-1], [c+'foo' for c in self.cnt.axes[-1]])


    def test_postfix_w_wrong_type(self):
        """Raise TypeError if postfix is neither None or str."""
        with self.assertRaises(TypeError):
            apply_spatial_filter(self.cnt, self.w, postfix=1)


    def test_apply_spatial_filter_copy(self):
        """apply_spatial_filter must not modify arguments."""
        cpy = self.cnt.copy()
        apply_spatial_filter(self.cnt, self.w)
        self.assertEqual(self.cnt, cpy)


    def test_apply_spatial_filter_swapaxes(self):
        """apply_spatial_filter must work with nonstandard chanaxis."""
        epo_f = apply_spatial_filter(swapaxes(self.epo, 1, -1), self.w,  chanaxis=1)
        epo_f = swapaxes(epo_f, 1, -1)
        epo_f2 = apply_spatial_filter(self.epo, self.w)
        self.assertEqual(epo_f, epo_f2)


if __name__ == '__main__':
    unittest.main()

