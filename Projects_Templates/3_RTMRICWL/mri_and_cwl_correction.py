#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:25:54 2018

@author: nfcontrol
"""



# import statements - all of these are packages within the rt environment
import time
from random import random as rand

import sys
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt


import re  # regular expressions
import pickle  # to save/load data
import dynarray  # a growing numpy array
import mne  # EEGLAB for python


# amplifier driver
sys.path.append("./mushu")
import libmushu

import pylsl
from pylsl import StreamInfo, StreamOutlet

import pdb

# import scipy
# from scipy imposrt signal

# from collections import deque  # a FILO list useful for plotting!

# the real-time signal filters:
from rtfilters import HPF, LPF, BPF, MR, CWL


nbchan=64  # you need to adjust to what your amp is producing!!
fs=5000
TR=1.950
trsamples=int(TR*fs)
# you need to change the # of channels here, most likely
hpf=HPF(f=1.0, fs=fs, order=3, nbchan=nbchan)
lpf=LPF(f=1.0, fs=fs, order=3, nbchan=nbchan)
bpf=BPF(f=[12.0, 15.0], fs=fs, order=3, nbchan=nbchan)
#mr=MR(trsamples=10000, N_thr=5, corr_thr = 0.995, forget=6)
mr=MR(trsamples=trsamples, N_thr=5, corr_thr = 0.995, forget=5, highpass=[3, 1.0, fs])



# make an 'amp' that reads in the data stream (LSL)
amp = libmushu.get_amp('lslamp')
amp.configure() # this sets up the LSL making us able to use it
# if you wish to change settings - look in mushu/libmushu/drivers/labstreaminglayer.py
# you can use the python API of labstreaminglayer to fix things there
# https://github.com/labstreaminglayer/liblsl-Python/tree/b158b57f66bc82230ff5ad0433fbd4480766a849


# make a new LSL stream to send data away:
# first create a new stream info (here we set the name to BioSemi,
# the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
# last value would be the serial number of the device or some other more or
# less locally unique identifier for the stream as far as available (you
# could also omit it but interrupted connections wouldn't auto-recover)
info = StreamInfo('Python', 'EEG', nbchan, fs, 'float32', 'corrected')

outlet = StreamOutlet(info)




amp.start()
while True:
    
    time.sleep(0.01)
    data, marker = amp.get_data()

    # it doesn't make sense to do stuff, if theere is no data
    if data.shape[0] > 0:
        data, marker = amp.get_data()
        
        data2=hpf.handle(data)
        data3=mr.handle(data2)
    
    
        # send it away - you'd need to figure out whether to seond data
        # or transposed data (i.e. data.T)
        # pdb.set_trace()
        outlet.push_chunk(data3)
        
        

