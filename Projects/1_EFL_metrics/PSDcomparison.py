def plot_psd_pb(filename, metafile, markerfile, *args, **kwargs):
    ''' 
    This function plots the PSD (in dB) of the frequency range of your choice.
    This function is to be used after EEG acquisition. The PSD compares two events 
    if you so choose (such as eyes open-eyes closed) or you could plot just one 
    event and look at the topographical distribution at a particular frequency on the scalp.
    
    Uses MNE-Python.
        
        
    Sample usage: plot_psd_pb('file.eeg', 'file.meta', 'file.marker', 
                                frequency_range=[9, 12], n_samples=500000, 
                                replay, n_fft=256, montage='biosemi64',
                                resample_freq=1000)
    
    NOTE: At present, this only works for comparing eyes open and eyes closed with the marker codes 201, 202, 203 and 204
    from the EEG acquisition using the EEG-fMRI Localiser.
    Additionally, function can output a raw array "rt_raw":
        raw = plot_psd_pb('file.eeg', 'file.meta', 'file.marker', 
                                frequency_range=[9, 12], n_samples=500000, 
                                replay, n_fft=256, montage='biosemi64',
                                resample_freq=1000)
    '''
    
    import struct
    import json
    import time
    import matplotlib
    try:
        matplotlib.use('QT5Agg')
    except:
        matplotlib.use('QT4Agg')
    import numpy as np
    import matplotlib.pyplot as plt
    import mne
    from mne.time_frequency import psd_welch
    import re
    
    import sys
    sys.path.append('../../mushu')
    sys.path.append('../../mushu/libmushu')
    
    import libmushu
    amp = libmushu.get_amp('replayamp')
    
    f = open(filename, 'r')
    raw = np.fromfile(f, dtype=np.float32)
    raw = raw.reshape(round(len(raw)/64),64).transpose()
    f.close()
    
    # get markers and meta
    fhj=open(metafile)
    meta=json.load(fhj)
    fhj.close()
    
    if 'n_samples' not in kwargs.keys():
        n_samples = 500000
    else:
        if kwargs['n_samples'] > len(raw):
            raise ValueError('Number of samples cannot be greater than the number of samples in the EEG file (%d)' % len(raw))
        n_samples = kwargs['n_samples']
    
    if 'montage' not in kwargs.keys():
        montage = mne.channels.read_montage(kind='biosemi64')
        
    else:
        montage = mne.channels.read_montage(kind=kwargs['montage'])
        
    with open(markerfile) as file:
        content = file.readlines()
        
    content = [x.strip() for x in content] 
    
    ## create the marker matrix
    
    ev_arr=[]
    for i, item in enumerate(content):
        out=re.split("[\s|S|T]+", item)
        if 'Sync Off' in item[1]:
            sample = int(float(out[0])/1000*meta['Sampling Frequency'])
            code = 250
        elif '201' in out[1]:
            sample = int(float(out[0])/1000*meta['Sampling Frequency'])
            code = int(out[1])
            ev_arr.append([sample, 0, code])
            
        elif '202' in out[1]:
            sample = int(float(out[0])/1000*meta['Sampling Frequency'])
            code = int(out[1])
            ev_arr.append([sample, 0, code])
            
        elif '203' in out[1]:
            sample = int(float(out[0])/1000*meta['Sampling Frequency'])
            code = int(out[1])
            ev_arr.append([sample, 0, code])
            
        elif '204' in out[1]:
            sample = int(float(out[0])/1000*meta['Sampling Frequency'])
            code = int(out[1])
            ev_arr.append([sample, 0, code])
        
    ch = montage.ch_names[:64]
    fs = meta['Sampling Frequency']
    eo1_s = ev_arr[0][0]
    eo1_f = ev_arr[1][0]
    ec1_s = ev_arr[2][0]
    ec1_f = ev_arr[3][0]
    eo2_s = ev_arr[4][0]
    eo2_f = ev_arr[5][0]
    ec2_s = ev_arr[6][0]
    ec2_f = ev_arr[7][0]
    m1 = raw[:, eo1_s:ec1_f]
    m2 = raw[:, eo2_s:ec2_f]
    m = np.hstack((m1,m2))
    m = np.array(m.transpose())
    
    ## replay file
    if 'replay' in args:
        
        amp.configure(m, (), ch, fs, realtime=True, blocksize_samples=20)
        amp.start()
        alld=[]
        allm=[]
        starttime=time.time()
        newtime=starttime
        i=0;
        while time.time() - starttime < n_samples/fs:

            while time.time() - newtime < 0.5:
                pass
            else:
                data, marker = amp.get_data()
        
                alld.append(data)
                for m in marker:
                    allm.append(marker)
                print(marker)
                
        
        

            print('%d' % i, end='', flush=True)
            i+=1
            newtime+=0.5
            
        amp.stop()
        m=np.concatenate(alld)
            
    if 'frequency_range' in kwargs.keys():
        if len(kwargs['frequency_range']) is not 2:
            raise ValueError('Frequency range length is %d, needs to be 2.' % len(frequency_range))
        frequency_range = kwargs['frequency_range']
    else:
        frequency_range = [2,25]
        
    fmin, fmax = frequency_range     
            
    
            
            
    #create info
    
    info = mne.create_info(
    ch_names = ch,
    ch_types = ['eeg' for i in range(len(ch))],
    sfreq = fs,
    montage = montage
    )
            
    ## get data
    rt = np.transpose(m)
    rt_raw = mne.io.RawArray(rt, info)
    
    # create marker channel for MNE python:
    if ev_arr:
        ev_arr[0][0] = 0
        ev_arr[1][0] = eo1_f - eo1_s
        ev_arr[2][0] = ec1_s - eo1_s
        ev_arr[3][0] = ec1_f - eo1_s - 10
        ev_arr[4][0] = ec1_f - eo1_s
        ev_arr[5][0] = (ec1_f - eo1_s) + (eo2_f - eo2_s)
        ev_arr[6][0] = (ec1_f - eo1_s) + (eo2_f - eo2_s) + (ec2_s - eo2_f)
        ev_arr[7][0] = (ec1_f - eo1_s) + (eo2_f - eo2_s) + (ec2_s - eo2_f) + (ec2_f - ec2_s) - 1
        print(ev_arr)
        stim_info = mne.create_info(['STI'], rt_raw.info['sfreq'], ['stim'])
        stim_data = np.zeros((1, len(rt_raw.times)))
        stim_raw = mne.io.RawArray(stim_data, stim_info)
        rt_raw.add_channels([stim_raw], force_update_info=True)

        # create the marker matrix:
        rt_raw.add_events(ev_arr)
        
    events = mne.find_events(rt_raw, initial_event=True)
    
    if 'resample_freq' in kwargs.keys():
        print('Resampling to %f, please be patient...' % kwargs['resample_freq'])
        rt_raw.resample(kwargs['resample_freq'], npad='auto')
        
    mne.set_eeg_reference(rt_raw)
    
    ## eyes open
    raw_eo1 = rt_raw.copy().crop(events[0][0]/info['sfreq'], events[1][0]/info['sfreq'])
    raw_eo2 = rt_raw.copy().crop(events[4][0]/info['sfreq'], events[5][0]/info['sfreq'])
    raw_eo = rt_raw.copy().crop(0, 0.1)
    raw_eo.append([raw_eo1, raw_eo2])

    ## eyes closed
    raw_ec1 = rt_raw.copy().crop(events[2][0]/info['sfreq'], events[3][0]/info['sfreq'])
    raw_ec2 = rt_raw.copy().crop(events[6][0]/info['sfreq'], events[7][0]/info['sfreq'])
    raw_ec = rt_raw.copy().crop(0, 0.1)
    raw_ec.append([raw_ec1, raw_ec2])
    

    # do psds
    if 'n_fft' not in kwargs.keys():
        n_fft = 16384
    else:
        n_fft = kwargs['n_fft']
        
    psd_eo, freqs_eo = psd_welch(raw_eo, fmin=fmin, fmax=fmax, n_fft=n_fft)
    psd_ec, freqs_ec = psd_welch(raw_ec, fmin=fmin, fmax=fmax, n_fft=n_fft)
    
    log_psd_eo = 10 * np.log10(psd_eo)
    log_psd_ec = 10 * np.log10(psd_ec)
    fig, ax = plt.subplots(1, 2)
    psds_mean_eo = log_psd_eo.mean(0)
    psds_mean_ec = log_psd_ec.mean(0)
    psds_diff = log_psd_ec - log_psd_eo
    psds_diff_mean = psds_diff.mean(0)
    x=[freqs_eo[int(np.where(psds_diff_mean == np.max(psds_diff_mean))[0][0]-np.std(psds_diff_mean).round())], 
       freqs_eo[int(np.where(psds_diff_mean == np.max(psds_diff_mean))[0][0]+np.std(psds_diff_mean).round())]]

    ## plot psds
    ax[0].plot(freqs_eo, psds_mean_eo, color='r', label='Eyes open')
    ax[0].plot(freqs_ec, psds_mean_ec, color='b', label='Eyes closed')
    ax[0].plot(freqs_eo, psds_diff_mean, color='g', label='Difference')
    ax[0].vlines(x,                                     
                 ymin=np.min(psds_diff_mean), ymax=np.max(psds_diff_mean)+5, colors='m',                                        
                 label=('Suggested frequency bounds for NF: {} Hz and {} Hz').format(round(x[0]), round(x[1])))                                                                                   
    ax[0].legend()
    ax[0].set(title='Welch PSD (EEG)', xlabel='Frequency',
              ylabel='Mean Power Spectral Density (dB)', 
              xticks=np.arange(round(np.min(freqs_ec)), round(np.max(freqs_eo)), step=2))

    ## plot topomap
    im, fl = mne.viz.plot_topomap(psds_diff.T[np.where(psds_diff_mean == np.max(psds_diff_mean))[0][0]],
                                  pos=info, 
                                  axes=ax[1], 
                                  vmin=-np.max(psds_diff.T[np.where(psds_diff_mean == np.max(psds_diff_mean))[0][0]]), 
                                  vmax=np.max(psds_diff.T[np.where(psds_diff_mean == np.max(psds_diff_mean))[0][0]])
                                 )
    cbar = fig.colorbar(im, ax=ax[1])
    cbarlabel = ('Power Spectral Density (dB) at {} Hz').format(freqs_eo[np.where(psds_diff_mean == np.max(psds_diff_mean))[0][0]])
    cbar.set_label(cbarlabel)
    fig.tight_layout()
    print('Suggested frequency bounds for NF: {} Hz and {} Hz'.format(round(x[0]), round(x[1])))
    if 'show_fig' in args:
        fig.show()
    return rt_raw