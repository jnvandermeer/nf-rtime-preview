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
sys.path.append("./mushu/libmushu/driver")

import libmushu
from labstreaminglayer import LSLAmp

import pylsl
from pylsl import StreamInfo, StreamOutlet

import ipdb
import pdb
import copy

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
mr=MR(trsamples=trsamples, N_thr=5, corr_thr = 0.995, forget=5, highpass=[])



# make an 'amp' that reads in the data stream (LSL)
amp = LSLAmp()
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
info = StreamInfo('MRCWL', 'data', nbchan, fs, 'double64', 'corrected')

#streams = pylsl.resolve_stream('type','EEG')
#info=pylsl.stream_inlet(streams[0]).info()
#info_out = copy.copy(info)



#info = StreamInfo('FromJupyter', 'EEG', 8, 100, 'float32', 'myuid2424')

# append some meta-data
#info.desc().append_child_value("manufacturer", "BioSemi")
#channels = info.desc().append_child("channels")
#for c in ["C3", "C4", "Cz", "FPz", "POz", "CPz", "O1", "O2"]:
#    channels.append_child("channel") \
#        .append_child_value("label", c) \
#        .append_child_value("unit", "microvolts") \
#        .append_child_value("type", "EEG")

# next make an outlet; we set the transmission chunk size to 32 samples and
# the outgoing buffer size to 360 seconds (max.)
# outlet = StreamOutlet(info, 32, 360)



outlet = StreamOutlet(info, 100, 30)




amp.start()
while True:
    
    # time.sleep(0.01)
    data, marker = amp.get_data()
    

    # it doesn't make sense to do stuff, if theere is no data
    if data.shape[0] > 0:
        #data, marker = amp.get_data()
        

        data2=hpf.handle(data)
        data3=mr.handle(data2)
    
    
        # send it away - you'd need to figure out whether to seond data
        # or transposed data (i.e. data.T)
        # pdb.set_trace()
        outlet.push_chunk(data.tolist())
        
        

