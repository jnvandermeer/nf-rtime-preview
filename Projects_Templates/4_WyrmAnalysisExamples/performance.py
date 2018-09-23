#!/usr/bin/env python

from __future__ import division


import time
import logging
import cPickle as pickle
import sys
import argparse

sys.path.append('../')

import numpy as np
from matplotlib import pyplot as plt

from wyrm import processing as proc
from wyrm.types import BlockBuffer, RingBuffer, Data

logger = logging.getLogger()


def online_erp(fs, n_channels, subsample):
    logger.debug('Running Online ERP with {fs}Hz, and {channels}channels'.format(fs=fs, channels=n_channels))

    target_fs = 100
    # blocklen in ms
    blocklen = 1000 * 1 / target_fs
    # blocksize given the original fs and blocklen
    blocksize = fs * (blocklen / 1000)


    MRK_DEF = {'target': 'm'}
    SEG_IVAL = [0, 700]
    JUMPING_MEANS_IVALS = [150, 220], [200, 260], [310, 360], [550, 660]
    RING_BUFFER_CAP = 1000

    cfy = [0, 0]

    fs_n = fs / 2

    b_l, a_l = proc.signal.butter(5, [30 / fs_n], btype='low')
    b_h, a_h = proc.signal.butter(5, [.4 / fs_n], btype='high')
    zi_l = proc.lfilter_zi(b_l, a_l, n_channels)
    zi_h = proc.lfilter_zi(b_h, a_h, n_channels)

    ax_channels = np.array([str(i) for i in range(n_channels)])

    names = ['time', 'channel']
    units = ['ms', '#']

    blockbuf = BlockBuffer(blocksize)
    ringbuf = RingBuffer(RING_BUFFER_CAP)

    times = []

    # time since the last data was acquired
    t_last = time.time()

    # time since the last marker
    t_last_marker = time.time()

    # time since the experiment started
    t_start = time.time()

    full_iterations = 0
    while full_iterations < 500:

        t0 = time.time()

        dt = time.time() - t_last
        samples = int(dt * fs)
        if samples == 0:
            continue
        t_last = time.time()

        # get data
        data = np.random.random((samples, n_channels))
        ax_times = np.linspace(0, 1000 * (samples / fs), samples, endpoint=False)
        if t_last_marker + .01 < time.time():
            t_last_marker = time.time()
            markers = [[ax_times[-1], 'm']]
        else:
            markers = []

        cnt = Data(data, axes=[ax_times, ax_channels], names=names, units=units)
        cnt.fs = fs
        cnt.markers = markers

        # blockbuffer
        blockbuf.append(cnt)
        cnt = blockbuf.get()
        if not cnt:
            continue

        # filter
        cnt, zi_l = proc.lfilter(cnt, b_l, a_l, zi=zi_l)
        cnt, zi_h = proc.lfilter(cnt, b_h, a_h, zi=zi_h)

        # subsample
        if subsample:
            cnt = proc.subsample(cnt, target_fs)
        newsamples = cnt.data.shape[0]

        # ringbuffer
        ringbuf.append(cnt)
        cnt = ringbuf.get()

        # epoch
        epo = proc.segment_dat(cnt, MRK_DEF, SEG_IVAL, newsamples=newsamples)
        if not epo:
            continue

        # feature vectors
        fv = proc.jumping_means(epo, JUMPING_MEANS_IVALS)
        rv = proc.create_feature_vectors(fv)

        # classification
        proc.lda_apply(fv, cfy)

        # don't measure in the first second, where the ringbuffer is not
        # full yet.
        if time.time() - t_start < (RING_BUFFER_CAP / 1000):
            continue

        dt = time.time() - t0
        times.append(dt)

        full_iterations += 1

    return np.array(times)



def plot():

    BLUE = "#268bd2"
    RED = "#d33682"
    BLACK = "#002b36"
    LGRAY = "#eee8d5"
    DGRAY = "#93a1a1"

    plt.figure(figsize=(8, 4))

    with open('results.pickle', 'rb') as fh:
        results = pickle.load(fh)

    ranges = []
    x, y = [], []
    for s, t in results:
        ranges.append(t.max() - t.min())
        y.append(t)
        x.append(s[12:])
    x = [50, 100, 500] * 6

    bp = plt.boxplot(y, labels=x, whis='range', widths=0.7)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylim(bottom=1, top=1000)
    plt.setp(bp['whiskers'], lw=2, ls='solid', c=BLUE)
    plt.setp(bp['medians'], lw=2, c=RED)
    plt.setp(bp['boxes'], lw=2, c=BLUE)
    plt.setp(bp['caps'], lw=2, c=BLUE)

    plt.ylabel('execution time [ms]')
    plt.xlabel('number of channels')
    plt.yscale('log')
    plt.grid(which='major', axis='y', ls='--', color=DGRAY)
    plt.grid(which='minor', axis='y', ls='-.', color=LGRAY)
    for i in range(5):
        plt.vlines((i+1)*3+.5, 0, 300, color=BLACK)
    plt.vlines(9.5, 0, 1000, color=BLACK, lw=3)
    plt.text(5, 600, 'with subsampling', color=BLACK, weight='bold', horizontalalignment='center')
    plt.text(14, 600, 'without subsampling', color=BLACK, weight='bold', horizontalalignment='center')
    for i, t in enumerate(['100Hz', '1kHz', '10kHz']):
        plt.text(i*3+2, 200, t, color=BLACK, horizontalalignment='center')
        plt.text(i*3+11, 200, t, color=BLACK, horizontalalignment='center')
    for i, r in enumerate(ranges):
        plt.text(i+1, 1.5, "{range:.1f}".format(range=r),
                 horizontalalignment='center', size='x-small', color=BLUE, weight='semibold')

    plt.tight_layout()
    plt.show()

def measure():
    target_fs = 100, 1000, 10000
    target_chans = 50, 100, 500

    results = []

    for subsample in 1, 0:
        for fs in target_fs:
            for chan in target_chans:
                t = online_erp(fs, chan, subsample=subsample)
                t *= 1000
                s = "{ss}subsampling\n{fs}Hz\n{chan} channels".format(ss=subsample, fs=fs, chan=chan)
                results.append((s, t))

    with open('results.pickle', 'wb') as fh:
        pickle.dump(results, fh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Measure online performance.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--measure', action='store_true')
    group.add_argument('--plot', action='store_true')

    args = parser.parse_args()
    if args.measure:
        measure()
    elif args.plot:
        plot()
