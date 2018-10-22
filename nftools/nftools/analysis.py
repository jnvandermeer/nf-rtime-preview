#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:08:23 2018

@author: johan
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
import copy
import pdb

# from this data (in memory), create a new MNE dataset.
def convert_alld_allm_to_mne(alld, allm, ch_names,s_freq):
    """ so if you just acquired a bunch of data -- use this to convert it to
    an MNE data structure, so you can run your analyses
    """
    
    # from this data (in memory), create a new MNE dataset.
    # fist the montage:
    
    montage=mne.channels.read_montage('standard_1005', ch_names=ch_names)    
    
    info = mne.create_info(
        #ch_names=['ch%d' % (i+1) for i in range(alld.shape[1])],
        ch_names=ch_names,
        ch_types=['eeg' for i in range(alld.shape[1])],
        sfreq=s_freq,
        montage=montage
    )
    
    raw=mne.io.RawArray(np.transpose(alld), info)

    # set reference to 0:
    # print('setting reference to []')
    # raw.set_eeg_reference(ref_channels=[])
    
    # handle the data:
    ev_arr=[]
    for i, item in enumerate(allm):
        sample = int(float(item[0])/1000*raw.info['sfreq'])
        
        mvalue = item[1]
        if isinstance(mvalue,str):
            if mvalue == 'boundary':
                pass
            else:
                mvalue = int(mvalue[1:])
        ev_arr.append([sample, 0, mvalue])
    
    
    print(raw.info['sfreq'])
    # print(raw.info['stim'])
        
    info = mne.create_info(['STI 014'], raw.info['sfreq'], ['stim'])  # this is actually correct
    stim_data = np.zeros((1, len(raw.times)))
    stim_raw = mne.io.RawArray(stim_data, info)
    raw.add_channels([stim_raw], force_update_info=True)
    
    # create the marker matrix:
    if len(allm)>0:
        raw.add_events(ev_arr)
    
    # set calibrations
    for ch in raw.info['chs']:
        if ch['kind'] == 2:
            ch['unit_mul'] = 0
            ch['range'] = 1.0
            ch['cal']=1.0
            ch['unit']=107
            ch['coil_type']=1
        if ch['kind'] == 3:
            ch['coil_type'] = 0
            ch['logno'] = 64
            ch['scanno']= 64
            ch['cal']= 1.0
            ch['range']= 1.0
            ch['unit']= -1
            ch['unit_mul']: 0.0


    # set av. reference, too:
    # raw.set_eeg_reference('average', projection=True)
    # we will do this later.
            
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
        



def plot_compare_two_spectra(raw_a_in, raw_b_in, freqs=[1, 25], n_fft=2048, n_overlap=512, chs_to_include=['all'], freq_lims_topoplot=[8, 12], pow_lims = [-10, 20]):
    """ This f calculates PSD and plot PSD of A vs PSD of B
    Allows you to specify certain channels too (or all)
    Usable to get some pictures for our T-F analysis, EO/EC, etc.
    """
    
    raw_a=copy.deepcopy(raw_a_in)
    raw_b=copy.deepcopy(raw_b_in)
    
    
   
    # print('hallo!')
    # prevent us plotting STIM channels in the spectrum, drop it if we find it somewhere:
    raw_a_chnames = [i['ch_name'] for i in raw_a.info['chs']]
    raw_b_chnames = [i['ch_name'] for i in raw_b.info['chs']]
    # getting rid of STI if found
    
    unwanteds = ['STI 014','ECG','ecg','EOG','eog']
    for item in unwanteds:
        if item in raw_a_chnames:
            raw_a.drop_channels([item])        
        if item in raw_b_chnames:
            raw_b.drop_channels([item])        



    # check calibration issues in rawa and rawb; rawa and raw b     
    mul_factor = 1.0
    if 1e-6 in [raw_a.info['chs'][0]['cal'], raw_a.info['chs'][0]['range']]:
        mul_factor = 1.0 / 1e-6
    
    for i, cal in enumerate([item['cal'] for item in raw_a.info['chs']]):
        if cal != 1.0:
            raw_a[i,:] = raw_a[i,:][0] * mul_factor
            
    for i, cal in enumerate([item['cal'] for item in raw_b.info['chs']]):
        if cal != 1.0:
            raw_b[i,:] = raw_b[i,:][0] * mul_factor
    
    
    
    
    # use the drop_channel method also on the other channels if found.
    # we do require raw_a and raw_b to have the same # of channels in the same order.
    if chs_to_include[0] == 'all':
        to_throw = []
        to_throw_i = []
        to_keep = raw_a.ch_names
        to_keep_i = range(len(raw_a.ch_names))
    else:
        to_throw = []
        to_throw_i = []
        to_keep = []
        to_keep_i = []
        
        for i, existing_ch in enumerate(raw_a.ch_names):
            if existing_ch in chs_to_include:

                to_keep.append(existing_ch)
                to_keep_i.append(i)
            else:
                to_throw.append(existing_ch)
                to_throw_i.append(i)

    

    


    # this produces a MATRIX of PSDs -- one for each channel!
    psd_a, f_a = mne.time_frequency.psd_welch(raw_a, fmin=freqs[0], fmax=freqs[1] , n_fft=n_fft, n_overlap=n_overlap)
    psd_b, f_b = mne.time_frequency.psd_welch(raw_b, fmin=freqs[0], fmax=freqs[1] , n_fft=n_fft, n_overlap=n_overlap)
    
    # take the log10 of it all:
    log_psd_a = 10 * np.log10(psd_a)
    log_psd_b = 10 * np.log10(psd_b)
    log_psd_diff = log_psd_b - log_psd_a
    
    print(chs_to_include)
    print(to_keep_i)
    
    #
    # data for plotting : the frequency graphs:
    #
    
    # here - apply to_keep or to_throw -- using "advanced slicing' of numpy.
    log_psd_a_mean_over_chs     = log_psd_a[to_keep_i,:].mean(0)  # this uses axis 0 to calculate mean, so over channels.
    log_psd_b_mean_over_chs     = log_psd_b[to_keep_i,:].mean(0)
    log_psd_diff_mean_over_chs  = log_psd_diff[to_keep_i,:].mean(0)
    
    
    # print(to_keep_i)
    # print(to_keep)
    # print(log_psd_a_mean_over_chs.shape)
    # print(log_psd_b_mean_over_chs.shape)
    
    # the plots:
    fig = plt.figure()
    pax = plt.subplot(121)

    pax.plot(f_a, log_psd_a_mean_over_chs, color='r', label='A')
    pax.plot(f_a, log_psd_b_mean_over_chs, color='b', label='B')
    # pax.plot(f_a, log_psd_diff_mean_over_chs, color=(0.4, 0.4, 0.4), label='B - A')
    
    # the difference we can see, so we won't plot it.
    # but we might put up a Patch with the Freq Limits!
    
    
    plt.ylim(pow_lims[0], pow_lims[1])
    pax.legend()
    pax.set(title='chs = %s; topolims = %s' % (chs_to_include, freq_lims_topoplot), xlabel='Frequency (Hz)',
          ylabel='PSD (dB)', 
          xticks=np.arange(round(np.min(f_a)), round(np.max(f_a)), step=2))

    
    #
    # data for plotting: the topo graph:
    #


    # print(freq_lims_topoplot)
    # print(f_a)
    # pdb.set_trace()
    
    # for 'advanced slicing' along the frequency direction (for the topo plot)
    if len(freq_lims_topoplot) == 2:
        f_indices = np.where(np.logical_and(f_a < freq_lims_topoplot[1], f_a > freq_lims_topoplot[0]))[0]
        
        pax.plot(np.ones((2,1))*freq_lims_topoplot[0],pax.get_ylim(),'m')
        pax.plot(np.ones((2,1))*freq_lims_topoplot[1],pax.get_ylim(),'m')
        
    elif len(freq_lims_topoplot) == 1:
        f_indices = np.where(abs(f_a-freq_lims_topoplot[0]) == min(abs(f_a-freq_lims_topoplot[0])))[0]

        pax.plot(np.ones((2,1))*freq_lims_topoplot[0],pax.get_ylim(),'m')
    
    
    mask_sel_topo = np.zeros((len(raw_a.info['chs']),1),'bool')
    mask_sel_topo[to_keep_i] = True
    
    # pdb.set_trace()
    topo_data = log_psd_a[:, f_indices].mean(1)
    pax=plt.subplot(322)
    im, fl = mne.viz.plot_topomap(topo_data, raw_a.info, axes=pax, names=raw_a.ch_names, show_names=False, mask=mask_sel_topo)
    im.set_clim(pow_lims[0], pow_lims[1])
    cbar = fig.colorbar(im, ax=pax)

    topo_data = log_psd_b[:, f_indices].mean(1)
    pax=plt.subplot(324)
    im, fl = mne.viz.plot_topomap(topo_data, raw_a.info, axes=pax, names=raw_a.ch_names, show_names=False, mask=mask_sel_topo)
    im.set_clim(pow_lims[0], pow_lims[1])
    cbar = fig.colorbar(im, ax=pax)

    # the diff:
    topo_data = log_psd_diff[:, f_indices].mean(1)
    pax=plt.subplot(326)
    im, fl = mne.viz.plot_topomap(topo_data, raw_a.info, axes=pax, names=raw_a.ch_names, show_names=False, mask=mask_sel_topo)
    # difflim is different:
    dscale=(max(pow_lims) - min(pow_lims))/3.0
    im.set_clim(-dscale, dscale) 
    cbar = fig.colorbar(im, ax=pax)
    
    
    # plt.figure()
    # mne.viz.plot_sensors(raw_a.info, show_names=True)
    # here, we might make the arrows (which channels we selected) in White..
    
    
    #cbarlabel = ('PSD between %f and %f Hz' % (freq_lims_topoplot[0], freq_lims_topoplot[1]))
    #cbar.set_label(cbarlabel)
    fig.tight_layout()
    fig.show()
    
    return topo_data
