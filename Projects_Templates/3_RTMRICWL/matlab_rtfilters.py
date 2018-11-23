# import scipy
import copy

from scipy import signal
from scipy.signal import lfilter_zi

import numpy as np
from scipy.stats.stats import pearsonr
import pdb


def make_mr():
	return MR(trsamples=9750, N_thr=5, corr_thr = 0.995, forget=5, highpass=[3, 1.0, 5000])



class RtFilter():
    def __init__(self, *args, **kwargs):
        if len(args)>0:
            print(args)
        if len(kwargs)>0:
            print(kwargs)
        
    
    def handle(self, data):
        return(data)
    
    
    

class HPF(RtFilter):
    """ keeps track of the zi for you"""
    def __init__(self, f=1.0, fs=500, order=3, nbchan=64):
        super(HPF, self).__init__(f=f, fs=fs, order=order, nbchan=nbchan)
        
        self.rtfilter = RtFilter()
        
        
        self.f=f
        self.fs=fs
        self.order=3

        self.b, self.a = signal.butter(order, 2*f/fs, btype='high', analog=False)
        zi = lfilter_zi(self.b, self.a)
        self.zi = np.tile(zi, (nbchan, 1)).T
        
    
    def handle(self, data):
        # do all kinds of stuff specific to this filter
        
        
        data, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi, axis=0)
        
        return self.rtfilter.handle(data)
        


        
class LPF(RtFilter):
    """ keeps track of the zi for you"""
    def __init__(self, f=125.0, fs=500, order=3, nbchan=64):
        super(LPF, self).__init__(f=f, fs=fs, order=order, nbchan=nbchan)
        
        self.rtfilter = RtFilter()
        
        
        self.f=f
        self.fs=fs
        self.order=3

        self.b, self.a = signal.butter(order, 2*f/fs, btype='low', analog=False)
        zi = lfilter_zi(self.b, self.a)
        self.zi = np.tile(zi, (nbchan, 1)).T
        
    
    def handle(self, data):
        # do all kinds of stuff specific to this filter
        
        
        data, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi, axis=0)
        
        return self.rtfilter.handle(data)
        

       
class BPF(RtFilter):
    """ keeps track of the zi for you"""
    def __init__(self, f=[12.0, 15.0], fs=500, order=3, nbchan=64):
        super(BPF, self).__init__(f=f, fs=fs, order=order, nbchan=nbchan)
        
        self.rtfilter = RtFilter()
        
        
        self.f=f
        self.fs=fs
        self.order=3

        self.b, self.a = signal.butter(order, [2*f[0]/fs, 2*f[1]/fs], btype='band', analog=False)
        zi = lfilter_zi(self.b, self.a)
        self.zi = np.tile(zi, (nbchan, 1)).T
        
    
    def handle(self, data):
        # do all kinds of stuff specific to this filter
        
        
        data, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi, axis=0)
        
        return self.rtfilter.handle(data)
        
       
        
        

        
        
class MR(RtFilter):
    """
    MR Filter -- you can specify
    trsamples   : int, in samples, which should be how many samples are in your TR
    N_thr       : int, in amount of volumes, which is how many volumes should be gathered before buffer is used to correct the data
    corr_thr    : threshold of correlation between artifact templates
    forget      : if artifact vary, this procedure will make a new artifact template waveform- it keeps the old one, though, but will forget them if it's not used for a long time (i.e., after 'forget' TRs)
    highpass    : In case you want to retain slow oscillations in the MR-corrected signal, this is where you specify order, freq and srate. Recommend you provide an empty list here and do LPF before doing MR.

    after initializing -- pass data using MR.handle(data) -- it should return MRI-cleaned data.

    """
    
    
    def __init__(self, trsamples=10000, N_thr=5, corr_thr = 0.995, forget=10, highpass=[3, 1.0, 5000]):
        super(MR, self).__init__(trsamples=trsamples, N_thr=N_thr, corr_thr=corr_thr, forget=forget, highpass=highpass)
        self.rtfilter = RtFilter()
        
        self.mrt=[]
        self.mrN=[]
        self.mrLastUsed=[]
        
        self.forget = forget
        
        self.selected=-1
        self.buff_initialized = False
        self.corr_thr = corr_thr
        self.N_thr = N_thr
        self.trsamples = trsamples    
        
        self.applyHPF = False
        if len(highpass)>0:
            # make the high-pass butter filter
            self.hpb, self.hpa = signal.butter(highpass[0], 2*highpass[1]/highpass[2], btype='high', analog=False)  # a high-pass filter
            self.hpzi = lfilter_zi(self.hpb, self.hpa)
            self.applyHPF = True
            self.zi_initiated = False

        
    def handle(self, datain):
        ''' actually do the mr correction - gather MR segments into a little list, and if certain conditions are met, 
            do a subtraction to get clean mr data. very easy template-based subtraction.
            In case of severe MR artifact changes (taken over all channels), this can also reset/change the buffer on the fly.
            This is handy for not having to stop the acquisition!
        '''
        
        
        nbpoints = datain.shape[0]
        
        # filter it - butterworth style:
        if self.applyHPF:
            # we consider the incoming data - but only filtered
            
            if not self.zi_initiated:
                self.hpzi = np.tile(self.hpzi, (datain.shape[1], 1)).T
                self.zi_initiated = True
            
            # pdb.set_trace()
            #data, self.hpzi = signal.lfilter(self.hpb, self.hpa, datain, zi=self.hpzi, axis=0)
            data, self.hpzi = signal.lfilter(self.hpb, self.hpa, datain, zi=self.hpzi, axis=0)
            #data = signal.filtfilt(self.hpb, self.hpa, datain, axis=0)
        else:
            data = datain


        
        
        
        # create the buffer without having to do it in __init__
        # so we can know the nbchan 
        if not self.buff_initialized:
            self.nbchan = data.shape[1]
            self.buff = np.zeros((self.trsamples, self.nbchan), dtype='float')
            self.buff_i = 0
            self.buff_initialized = True
            
            
            


            
        
        
        # finish the current template/buffer, or do the work
        # to consider a new MR template?
        if self.buff_i + nbpoints < self.trsamples:
            
            # store data in buffer
            self.buff[self.buff_i:self.buff_i+nbpoints,:] = data
            
            
            # can we correct it? -- then this is >0
            if self.selected >= 0:
                # look up what kind of MR we need to remove from library
                t_d = self.mrt[self.selected]
            
                #pdb.set_trace()
                corr_data = datain - t_d[self.buff_i:(self.buff_i+nbpoints),:]
            
            else:
                corr_data = datain
            
            self.buff_i += nbpoints
            return self.rtfilter.handle(corr_data)
            
            
        else:

            #print(self.mrN)
            #print(self.mrLastUsed)
            
            prev_selected = self.selected
            
            # complete buffer
            self.buff[self.buff_i:self.trsamples,:] = data[0:(self.trsamples-self.buff_i),:]
            
            # temporary buffer to be used for later...
            tbuff = data[(self.trsamples-self.buff_i):nbpoints,:]
            
            
            
            # this will happen also with the followiung code, anyway.
            #            # is there some info yet about MR artifacts? if not - create..
            #            if len(self.mrt) == 0:
            #                self.mrt.append(np.zeros((self.trsamples, self.nbchan), dtype='float'))
            #                self.mrN.append(1)
            #                self.mrLastUsed.append(1)
            #
            #                # and store buff in that, too                
            #                self.mrt[-1] = self.buff


            # calculate all of the correlations with the rest of them
            # and how many are in there, too...
            #corrs, N = zip(*[(pearsonr(self.buff.flatten(), m.flatten()), self.mrN(i_N)) for i_N, m in enumerate(self.mrt)])
            #if len(self.mrN)>0:
            #    if self.mrN[0]>5:
            #        pdb.set_trace()
            corrs = [pearsonr(self.buff.flatten(), m.flatten())[0] for m in self.mrt]
            #print(corrs)

            # update the library -- actions
            if any([corr > self.corr_thr for corr in corrs]):

                # select what our current marked library of templates is:
                i_maxcorr = corrs.index(max(corrs))
    
                # add it to the select library:
                # weighting 1 : restore original values
                self.mrt[i_maxcorr] *= float(self.mrN[i_maxcorr])
    
                # weighting 2 : increase the N
                self.mrN[i_maxcorr] += 1.0
    
                # weighting 3 : divide with the new N
                self.mrt[i_maxcorr] /= self.mrN[i_maxcorr]
    
                # then add the new buff; weighted:
                self.mrt[i_maxcorr] += copy.copy(self.buff) / self.mrN[i_maxcorr]
                
                
                # HERE -- I should arrange that lastused is taken care of... : i_maxcorr
                # add 1 to all last N, but set the one of self.selected to 1 (i.e., this is the last used one)
                self.mrLastUsed = [*map(lambda x: x+1, self.mrLastUsed)]
                self.mrLastUsed[i_maxcorr] = 1
    
            else:
                
                # HERE -- taking care : the last one!
    
                # so - the buffer is completely NEW -- so let's try making a new template, then - + store it inside
                self.mrt.append(np.zeros((self.trsamples, self.nbchan), dtype='float'))
                #self.mrt.append(self.buff)
                self.mrN.append(1)
                self.mrLastUsed.append(0)
                # and store it, too
                self.mrt[-1] = copy.copy(self.buff) # HERE -- we add the buffer!!
                corrs.append(0)  # also grow corrs - for later:


                # HERE -- I should arrange that lastused is taken care of... : i_maxcorr
                # add 1 to all last N, but set the one of self.selected to 1 (i.e., this is the last used one)
                self.mrLastUsed = [*map(lambda x: x+1, self.mrLastUsed)]
                self.mrLastUsed[-1] = 1


            # figure out which one we're going to select now: -- or NOT...
            keepi=[]
            keepcorr=[]
            keepN=[]
            for i, (c, N) in enumerate(zip(corrs, self.mrN)):
                if N > self.N_thr and c > self.corr_thr:
                    keepi.append(i)
                    keepcorr.append(i)
                    keepN.append(i)
             
            # do we select anything now?
            if len(keepi) > 0:
        
                #print('----')
                #print(keepi)
                #pdb.set_trace()
                # so these all adhere to the thresholds -- now figure out which one to choose for artifact correction -- and update 'self.selected'
                self.selected = keepi[keepcorr.index(max(keepcorr))]
                #print('selected: %d' % self.selected)

            else:
                self.selected = -1                

            
            # correct data -- part I -- from the old buffer
            # correct the prev buffer with prev_selected
            if prev_selected >= 0:
            
                t_d = self.mrt[prev_selected]
            
                tosubtract1 = t_d[self.buff_i:self.buff_i+self.trsamples,:]
            else:
                tosubtract1 = np.zeros((self.trsamples-self.buff_i, self.nbchan), dtype='float')
        
            if self.selected >= 0:
                
                t_d = self.mrt[self.selected]
                
                tosubtract2 = t_d[0:(-1*(self.trsamples-self.buff_i)+nbpoints),:]
            else:
                tosubtract2 = np.zeros((-1*(self.trsamples-self.buff_i)+nbpoints, self.nbchan), dtype='float')
                # correct data -- part II -- from the new buffer
                # correct the current buffer with NEW selected
                
            tosubtract_full = np.concatenate((tosubtract1, tosubtract2))
            # get rid of the very old stuff in library
            
            
            # update the buffer:
            #pdb.set_trace()
            self.buff[:,:]=0
            self.buff_i = -1*(self.trsamples-self.buff_i)+nbpoints
            self.buff[0:self.buff_i,:] = tbuff
            

            
            # obselete code:
            # corr_data = data - tosubtract_full



        #first throw things away...
        #    pdb.set_trace()
        # forget the old stuffm after self.forget times of not-being-used
        mark=[i for i, lastused in enumerate(self.mrLastUsed) if lastused > self.forget]
        
        # forget them...
        for popi in reversed(mark): #[::-1]:
            self.mrt.pop(popi)
            self.mrN.pop(popi)
            self.mrLastUsed.pop(popi)  
        # data=copy.copy(datain) -- making sure that our selected remains pointed
        # to the right one
        if len(mark)>0 and self.selected>=0:
            #print('----')
            #print(self.selected)
            self.selected -= sum([self.selected > i for i in mark])
            #print(self.selected)
            #print(mark)
            
        # handle return...
        return self.rtfilter.handle(datain - tosubtract_full)