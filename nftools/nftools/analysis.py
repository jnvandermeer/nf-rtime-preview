#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:08:23 2018

@author: johan
"""
import mne
import numpy as np

# from this data (in memory), create a new MNE dataset.
def convert_alld_allm_to_mne(alld, allm, channels):
    """ so if you just acquired a bunch of data -- use this to convert it to
    an MNE data structure, so you can run your analyses
    """
    
    # from this data (in memory), create a new MNE dataset.
    # fist the montage:
    
    montage=mne.channels.read_montage('standard_1005', ch_names=channels)    
    
    info = mne.create_info(
        #ch_names=['ch%d' % (i+1) for i in range(alld.shape[1])],
        ch_names=channels,
        ch_types=['eeg' for i in range(alld.shape[1])],
        sfreq=1000,
        montage=montage
    )
    
    raw=mne.io.RawArray(np.transpose(alld), info)
    
    # handle the data:
    ev_arr=[]
    for i, item in enumerate(allm):
        sample = int(float(item[0])/1000*raw.info['sfreq'])
        ev_arr.append([sample, 0, item[1]])
    
    
        
    info = mne.create_info(['STI 014'], raw.info['sfreq'], ['stim'])
    stim_data = np.zeros((1, len(raw.times)))
    stim_raw = mne.io.RawArray(stim_data, info)
    raw.add_channels([stim_raw], force_update_info=True)
    
    # create the marker matrix:
    raw.add_events(ev_arr)
    
    # set calibrations
    for ch in raw.info['chs']:
        if ch['kind'] == 2:
            ch['cal']=1e+6
            
    return raw


def select_part_from_mne_dataset(rawin, **kwargs):
    """ This function should allow you to make a new data set from an MNE dataset
    where you select b and e as samples, and all evs, etc, will be handled.
    Basically, you have to make a new dataset from the old MNE dataset.
    I don't know why a function like this doesn't already exist.
    """
    
    # print('hallo2!!!')
    if 'boundaries' in kwargs.keys():
        print('using boundaries')
        boundaries=kwargs['boundaries']
        new_info=rawin.info
        
        newdat=[]
        if isinstance(boundaries[0], list) or isinstance(boundaries[0], tuple):
            for item in boundaries:
                b, e = item
                newdat.append(rawin[:,b:e][0])
        else:
            b=boundaries[0]
            e=boundaries[1]
            
            newdat.append(rawin[:,b:e][0])
        
        newdat=np.concatenate(newdat, axis=1)
                
        print(newdat)
        newraw=mne.io.RawArray(newdat, new_info)
        
        return(newraw)
    
    elif 'markers' in kwargs.keys():
        print('using markers')
        markers = kwargs['markers']
        if len(markers) != 2:
            raise Exception('Provide 2 markers for this operation (numbers!)')

        bmarker=markers[0]
        emarker=markers[1]
        
        # so - what are all the events??
        
        evs=mne.find_events(rawin)
        allbs=[st[0] for st in evs if st[2] == bmarker]
        alles=[st[0] for st in evs if st[2] == emarker]
        
        if len(allbs) != len(alles):
            raise Exception('cannot find nice segments')
            
        newraw = select_part_from_mne_dataset(rawin, boundaries=list(zip(allbs, alles)))
        
        return(newraw)
        # find the markers from 'b':
        

