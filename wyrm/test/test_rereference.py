from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import rereference
from wyrm.processing import swapaxes


CHANS = 5
SAMPLES = 20
EPOS = 3


class TestRereference(unittest.TestCase):

    def setUp(self):
        dat = np.zeros((SAMPLES, CHANS))
        # [-10, -9, ... 20)
        dat[:, 0] = np.arange(SAMPLES) - SAMPLES/2
        channels = ['chan{i}'.format(i=i) for i in range(CHANS)]
        time = np.arange(SAMPLES)
        self.cnt = Data(dat, [time, channels], ['time', 'channels'], ['ms', '#'])
        # construct epo
        epo_dat = np.array([dat + i for i in range(EPOS)])
        classes = ['class{i}'.format(i=i) for i in range(EPOS)]
        self.epo = Data(epo_dat, [classes, time, channels], ['class', 'time', 'channels'], ['#', 'ms', '#'])

    def test_rereference_cnt(self):
        """Rereference channels (cnt)."""
        cnt_r = rereference(self.cnt, 'chan0')
        dat_r = np.linspace(SAMPLES/2, -SAMPLES/2, SAMPLES, endpoint=False)
        dat_r = [dat_r for i in range(CHANS)]
        dat_r = np.array(dat_r).T
        dat_r[:, 0] = 0
        np.testing.assert_array_equal(cnt_r.data, dat_r)

    def test_rereference_epo(self):
        """Rereference channels (epo)."""
        epo_r = rereference(self.epo, 'chan0')
        dat_r = np.linspace(SAMPLES/2, -SAMPLES/2, SAMPLES, endpoint=False)
        dat_r = [dat_r for i in range(CHANS)]
        dat_r = np.array(dat_r).T
        dat_r[:, 0] = 0
        dat_r = np.array([dat_r for i in range(EPOS)])
        np.testing.assert_array_equal(epo_r.data, dat_r)

    def test_raise_value_error(self):
        """Raise ValueError if channel not found."""
        with self.assertRaises(ValueError):
            rereference(self.cnt, 'foo')

    def test_case_insensitivity(self):
        """rereference should not care about case."""
        try:
            rereference(self.cnt, 'ChAN0')
        except ValueError:
            self.fail()

    def test_rereference_copy(self):
        """rereference must not modify arguments."""
        cpy = self.cnt.copy()
        rereference(self.cnt, 'chan0')
        self.assertEqual(self.cnt, cpy)

    def test_rereference_swapaxes(self):
        """rereference must work with nonstandard chanaxis."""
        dat = rereference(swapaxes(self.epo, 1, 2), 'chan0', chanaxis=1)
        dat = swapaxes(dat, 1, 2)
        dat2 = rereference(self.epo, 'chan0')
        self.assertEqual(dat, dat2)


if __name__ == '__main__':
    unittest.main()

