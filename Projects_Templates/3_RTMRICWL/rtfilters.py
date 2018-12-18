# import scipy
import copy
import multiprocessing

from scipy import signal
from scipy.signal import lfilter_zi


import scipy
import scipy.io


import numpy as np
from scipy.stats.stats import pearsonr
import pdb
import ipdb

import logging
import time
# import copy

# apparently, we can do this:
log=logging.getLogger('rtfilters')




# helpful snippet for making tons of matfiles (to be used later)
from time import mktime, gmtime

def now_matfile(): 
   return str(int(mktime(gmtime())))


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
        self.order=order

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
        self.order=order

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
        self.order=order

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
            




def estimate_it(X, y, Yprev, prev_betas):
    """ The function that ACTUALLY estimates beta's -- the rest is all upkeep & maintenance
    """
    
    # ipdb.set_trace()
    inv = np.linalg.pinv(np.dot(X.T, X))
    
    betas = np.dot(inv, np.dot(X.T, y))
    
    diff1 = np.dot(X, prev_betas) - np.dot(X, betas)
    diff2 = Yprev - np.dot(X, betas)


    
    
    fname = 'glm' + now_matfile()
    
    scipy.io.savemat(fname, {'X':X, 'betas':betas, 'prev_betas':prev_betas, 'diff1':diff1, 'diff2':diff2, 'y':y, 'Yprev':Yprev})
    #qdiff.put(diff)

    return betas, diff1, diff2

    # from these .mat files --> reconstruct the difference between 'optimal non-real-time' and 'use previous' and 'kalmanned'


class estimatorProcess(multiprocessing.Process):

    """ starting up a separate Process each time a window is to be estimated takes up too much computational resources
        so start up that Process just once - at the beginning of the meausrement
    """
    
    def __init__(self, q_in, q_out):
        
        super(estimatorProcess, self).__init__()
        self.ev_stop = multiprocessing.Event()
        self.q_in = q_in
        self.q_out = q_out
        
        
    
    def run(self):
        
        while not self.ev_stop.is_set():
            time.sleep(0.005)
            
            if not self.q_in.empty():
                
                X, y, Yprev, prev_betas = self.q_in.get()
                
                betas, diff1, diff2 = estimate_it(X, y, Yprev, prev_betas)
        
                self.q_out.put((betas, diff1, diff2))
        

    def stop(self):
        self.ev_stop.set()


            

class CWL(RtFilter):
    ''' This handles the CWL regression! -- I should also include optional HP filter here
        low-pass filtering should be done separately
    '''
    
    # some reasonable initial values
    def __init__(self, seconds_in_window=6.0, tdelay=0.035, icws=[1,2,3], ichs=[33,34,35,36], fs=1000, highpass=[]):
        super(CWL, self).__init__(seconds_in_window=seconds_in_window, tdelay=tdelay, icws=icws, ichs=ichs, fs=fs, highpass=highpass)
        
        self.rtfilter = RtFilter()
        
        
        self.DEBUG=False
        self.seconds_in_window=seconds_in_window  # how long (samples) should the window be that we use
        
        self.tdelay=tdelay  #=tfuture  # how many samples 'future' to use (output signal will be delayed by so many samples!)
        
        self.icws=icws  # which are the channels?
        self.ichs=ichs  # what are the noisy regressors (containing NO EEG)?
        
        self.fs=fs  # sampling rate
        
        # we set these without being able to change these!
        self.taperfunction = np.hanning  # windowing function to use
        self.taperfactor = 1  # how many overlapping windows?
        
        
        # useful data structures:
        self.nwin = round(seconds_in_window * fs)
        if self.nwin % 2 == 0:
            self.nwin += 1
        
        self.sfuture = round(tdelay * fs)
        self.spast = round(tdelay * fs)
        
        # this will be very useful later on!!
        self.delayvec = np.array(range(self.sfuture+self.spast+1)) - self.spast
        
        self.zerodelayindex = np.where(self.delayvec == 0)[0][0]
        
        self.XnTemplate = np.zeros((self.nwin, len(self.icws) * (self.sfuture+self.spast+1)))
        
        # calculate the windowing functions:
        m=np.concatenate((np.matrix(self.taperfunction(self.sfuture+self.spast+1)).T,np.matrix(range(self.sfuture+self.spast+1)).T),axis=1)
        
        #pdb.set_trace()
        m=np.tile(m,(len(self.icws),1))
        h=m[np.argsort(m[:,1],axis=0),0]
        
        # hanning-windowed delay in time-delay direction (over ~50 msec)
        hDelayMat=np.ones((self.nwin,1)) * h.T
        
        # ipdb.set_trace()
        
        # hanning-windowed delay in time direction (over ~6sec)
        hTimeMat=np.matrix(self.taperfunction(self.nwin)).T * np.ones((1,len(self.icws) * (self.sfuture+self.spast+1)))
        
        # the full window...
        self.hwts = np.array(np.multiply(hDelayMat, hTimeMat))
        
        # we need for y:
        self.hwy = np.array(self.taperfunction(self.nwin)).reshape(self.nwin, 1)

        
        self.taperlist = []   # this is where we collect stuff as per my notes.
        
        
        
        win_startstep = (self.nwin-1)/self.taperfactor/2
        if win_startstep % 2 != 0:
            raise Exception('Your window size is bad for current tapering specs')
            # the program stops here.
            # make sure that the tapering and window size are integer-divisible
            # otherwise things don't add up so nicely.
        else:
            win_startstep = int(win_startstep)
            
        
        for i in range(self.taperfactor*2):
            self.taperlist.append({
                    's':i * win_startstep, 
                    'Xn':np.zeros((self.nwin, len(self.icws) * (self.sfuture+self.spast+1))), 
                    'y':np.zeros((self.nwin,len(self.ichs))),
                    'Y':np.zeros((self.nwin,len(self.ichs))),
                    'b':np.zeros((len(self.icws)*(self.spast+self.sfuture+1),1))
                    # we don't do that, yet - this will remain 'open' always.
                    #'b':np.zeros((self.icws * (self.sfuture+self.spast+1), 1)),
                    #'number':0,
                    #'estimated': False
                    })
            
            
        
        self.s = 0  # this counts which point we are at currently!
        
        
        self._olddata = []  # to help us deal with delayed version of signals
        self._oldsignals = []
        self._oldcwls = []
        self._buffereddata = []
        self._buffereddata2 = []
        
        self._queue_estimate_it = multiprocessing.Queue()
        self._queue_incoming_betas = multiprocessing.Queue()

        # ipdb.set_trace()
        # start up our estimator Process:        
        self.estimatorProcess = estimatorProcess(self._queue_estimate_it, self._queue_incoming_betas)
        self.estimatorProcess.start()

        
        # self._queue_incoming_diffs = multiprocessing.Queue()
        self.betas_are_estimated=False
        self.betas=[np.zeros((len(self.icws)*(self.spast+self.sfuture+1),1))]
        
        self.currentbetas = self.betas[-1]
        self.switch_betas=False
        
        
    
    def handle(self, data):
        
        
        # first to do timeshifting stuff --> then worry about allocation into different subregions...
        ddata, dy, dXn = self._handle_delays(data)
        
        s=self.s  # current sample
        N=data.shape[0]
        nwin=self.nwin
        
        #cleaned_chs=[]
        # cleaned_data = self._return_delayed_data()
        # ipdb.set_trace()
        cleaned = []
        for i in range(len(self.taperlist)):
            cleaned.append(data[:,0:len(self.ichs)].copy())


        if self.DEBUG:
            sumcheck = []
            for i in range(len(self.taperlist)):
                sumcheck.append(data[:,0:len(self.ichs)].copy())
        

        # cwls = data[:,self.icws]  # better do it here - breakdown into cwls and signals
        # signals = data[:,self.ichs]  # when we return data, we replace some cols from data with corrected signals
        
        # go through the list as per our notes -- only COLLECT here, and start new estimations where necessary.
        for itaper, tli in enumerate(self.taperlist):
            
            finished=False
            cur_s = s
            
            
            # if itaper == 1 and tli['s'] > 9000 and cur_s > 17000:
            #     ipdb.set_trace()
            # re-do this -- figure out DATA indices, and X indices
            # then make a list of that
            # then just go through the list
            # otherwise I cannot follow it anymore
            
            while not finished:
                
                if cur_s >= tli['s']:
                    # ipdb.set_trace()
                    # check for beta queue                
                    if s+N > tli['s'] + nwin:
                        curN = tli['s'] + nwin - cur_s
                    else:
                        curN = s + N - cur_s
                
                        
                    bXn = cur_s - tli['s']  # see notes
                        
                    eXn = bXn + curN   # see notes
                    
                    Xindices=range(bXn,eXn)  # taking care of neg/positive...
                    
                    bd = cur_s - s   # see notes
                    ed = bd + curN   # see notes
                    
                    dindices=range(bd,ed)
                    # put y and X where they belong
                    #ipdb.set_trace()
                    
                    #pd = part of delay(ed)...
                    pdXn, pdy = self._handle_partition_and_hanning(dy, dXn, dindices, Xindices)

                    pdY = np.dot(pdXn, tli['b'])
                    pde = pdy-pdY

                    tli['Xn'][Xindices,:] = pdXn
                    tli['y'][Xindices,:] = pdy
                    tli['Y'][Xindices,:] = pdY
                
                    cleaned[itaper][dindices,:] = pde  # assign the data
                    if self.DEBUG:
                        sumcheck[itaper][dindices,:] = self.hwy[Xindices,:]
        
                if tli['s'] + nwin > s + N:
                    finished=True
                    gotoNextWindow=False
                elif tli['s'] + nwin == s + N:
                    finished=True
                    gotoNextWindow=True
                else:
                    finished=False
                    gotoNextWindow=True
                        
                        
                if gotoNextWindow:       
                 
                    # 'send off' the current window for estimation
                    self._estimate_betas(tli)
                    # ipdb.set_trace()
                    
                    # 'init' a new one, too. associate our most current beta's to that
                    self._check_switch_betas()  # figure out / upkeep on beta estimations - check the queue. -- kalman should be implemented here?
                    tli['b'] = self.betas[-1]   # assign the latest one(s) -- or the Kalman estimated ones

                    tli['s'] += nwin-1
                    cur_s = tli['s']
                    tli['Xn'] =np.zeros((self.nwin, len(self.icws) * (self.sfuture+self.spast+1)))
                    # you completed a window --> adjust s and N
                    

                    #ipdb.set_trace()
                    
                        
                    
                    # HERE - we figure out whether we 'set' the beta's - or not
                    # if they're set -- then, we we associate the latest set of beta's to the current window
                    # we also will not anymore check whether beta's are estimated
                    # instead, we fill the list of 'current' beta's with 0. That will produce 0, too.
                    # we also need to pass the current beta's...
                    
                    # if we're here, then we need to estimate something!
                    # send (complete) window to estimator
                    # ipdb.set_trace()
                    # self._check_switch_betas()

        #if s > 17000:
        #    ipdb.set_trace()

        # sum it all up
        if self.taperfactor == 1:
            cleaned_channels = sum(cleaned)  # summate over the tapers (we have 2 tapers with taperfactor == 1, usually.)
            if self.DEBUG:
                checked_sumcheck = sum(sumcheck)
        else:
            cleaned_channels = sum(cleaned) / float(self.taperfactor)
            if self.DEBUG:
                checked_sumcheck = sum(sumcheck) / float(self.taperfactor)

        
        #if s > 17000:
        #    ipdb.set_trace()

        # set the corrected channels to the channels that we just corrected!
        ddata[:,self.ichs] = cleaned_channels
        if self.DEBUG:
            ddata[:,self.ichs] = checked_sumcheck  # check whether the weights are always == 1!
        self.s += data.shape[0]   
        return self.rtfilter.handle(ddata)
                            


    def _handle_partition_and_hanning(self, dy, dXn, dindices, Xindices):
        
        """ So this should be simple enough 
            - use dincides on data(selecting a part of it)
            - then apply hanning window following Xindices on both dXn and dy
            - and return result?
        """
        
        
        # ipdb.set_trace()
        hanning_for_y = self.hwy[Xindices,:]
        
        pdy = dy[dindices] * hanning_for_y

        hanning_for_dXn = self.hwts[Xindices,:]
        
        pdXn = dXn[dindices,:] * hanning_for_dXn
        
        return pdXn, pdy
        
        
        
        
        

    def _handle_delays(self, data):
        """ This function return in delayed form: 
                (a) the full data (because we need it)
                (b) the data specified by the cwl's (i.e., y)
            reurns in time-expanded form:
                (c) the Xn - with the temporal expansion(s)
        """
        
        
        # first - buffer it
        self._buffer_data(data, self.sfuture + self.spast)
        
        
        # then get the data (and make a copy of it?)
        tmpdata = copy.deepcopy(self._buffereddata)
        
        # then: construct the Xn
        Xn = np.zeros((data.shape[0], len(self.icws) * (self.sfuture+self.spast+1)))

        # here is the magic where things get 'delayed' - the stuff we put in...
        for i, d in enumerate(self.delayvec):
            
            # this is now actually correct!
            startind = -len(self.delayvec)+i-data.shape[0]+1
            stopind = startind + data.shape[0]
            
            # ipdb.set_trace()
            tmpdata2=tmpdata[range(startind,stopind), :]
            Xn[:, i*len(self.icws):(i+1)*len(self.icws)] = tmpdata2[:, self.icws]        
        
        
        # Xn is done: now deal with y
        # normal_range:
        startind = -self.zerodelayindex-data.shape[0]
        stopind = startind + data.shape[0]
        
        alldata = tmpdata[range(startind,stopind),:]
        
        y=alldata[:,self.ichs]
        
        
        return alldata, y, Xn
        
            

    
    #    def _handle_delay_taper_cwl_channel_selection(self, data, windowing, betas):
    #        """ So, this will recall the last bit of the data, and take care
    #            of all of the tdelay stuff, so I don't have to worry about it
    #            in the main function
    #            Estimate new beta's in a separate Process.
    #            Will also multiply with the Taper Windows...
    #            and will also tease out the data channels (y) from the cwl channels(Xn)
    #            upon calling it -- it will store some old data to allow good time shifts...
    #            # we already buffered the data?
    #        """
    #
    #        # print(data.shape)
    #
    #        #if len(data)>0 and len(windowing)==0:
    #        #    ipdb.set_trace()
    #
    #        # the +1 is rather essential here.
    #        # the len of delayvec is f.e. 71 - but the delay of ==0
    #        # is already taken care of and doesn't require much adjustments 
    #        # therefore, we only need the stuff WITHOUT the +1
    #
    #        # self._buffer_data(data, self.sfuture + self.spast + 1)
    #        self._buffer_data(data, self.sfuture + self.spast)
    #        
    #        # this function should return the following items:
    #        # Xn - which is the part of the Design Matrix
    #        # y - which is the measured data (i.e. DELAYED data) - and coupled to Xn for (later) estimation of Betas
    #        # Y - this hs the (current) DM multiplied by beta wts
    #        # e - this is the cleaned EEG cor those channels that were selected
    #
    #        cwls=copy.copy(self._buffereddata[:,self.icws])
    #        signals = copy.copy(self._buffereddata[:,self.ichs])
    #
    #
    #        # ipdb.set_trace()
    #        # create the part of the Design Matrix now... - initialization...
    #        Xn = np.zeros((data.shape[0], cwls.shape[1] * (self.sfuture+self.spast+1)))
    #
    #
    #        # so let's see if this holds anything true...
    #        # ipdb.set_trace()        
    #        # this fills (not entire, but part of) matrix up with the CWL (time-delayed versions) of the signal
    #        # here is the magic where things get 'delayed' - the stuff we put in...
    #        for i, d in enumerate(self.delayvec):
    #            
    #            # this is now actually correct!
    #            startind = -len(self.delayvec)+i-data.shape[0]+1
    #            stopind = -len(self.delayvec)+i+1
    #            
    #
    #            Xn[:, i*len(self.icws):(i+1)*len(self.icws)] = cwls[range(startind,stopind),:]
    #
    #        # apply the windowing (passed on as argument...)
    #        Xn *= windowing
    #
    #        # and this is the y that goes with it:
    #        y = signals[-self.zerodelayindex - data.shape[0]:-self.zerodelayindex,:]
    #        
    #        
    #        # find the middle index (i.e., the 'pure' hanning window part of this guy)
    #        middleIndex=int((len(self.delayvec) - 1)/2 * len(self.icws))
    #        
    #        # expand it for all the y (i.e., the data)
    #        hann_multiplier_for_y = np.dot(windowing[:,middleIndex], np.ones((1, signals.shape[1])))
    #        
    #        # multiply it!!!
    #        y *= hann_multiplier_for_y
    #
    #        # this will be 0 initially, but as beta's are estimated and passed on -- will be nonzero (and artifact correction will kick in)
    #        Y = np.dot(Xn, betas)
    #
    #        # so this should be the artifact-corrected EEG signal - return it.
    #        e = y - Y
    #
    #        return Xn, y, Y, e
    #        # l['Xn'][bXn:eXn,:], l['y'][bXn:eXn,:], Y, e
        


    
    def _estimate_betas(self, tli):
        """ This function will actually Estimate beta's, from a 'completed' window l
        """
        # first it will make a copy that we're going to use
        
        # then it will make a Process that will do the actual estimation
        
        #ipdb.set_trace()
        X = tli['Xn']
        y = tli['y']
        Yprev = tli['Y']
        

        prev_betas = self.betas[-1]

        # contains 6 regressors; taken with different time delays
        # so the beta's are per cwl-per-delay the fit to y.
        # ipdb.set_trace()
        #p=multiprocessing.Process(target=estimate_it, args=(X, y, Yprev, self._queue_incoming_betas, prev_betas))
        #p.start()
        
        self._queue_estimate_it.put((X, y, Yprev, prev_betas))
        # self.processes.append(p)  # for joining later on...
            


            
    def _check_switch_betas(self):
        """ check the queues -- do upkeep with those outputs, plz...
        """
        # check the queus - if they are full/filled, then change betas
        #ipdb.set_trace()
        if not self._queue_incoming_betas.empty():
            betas, diff1, diff2 = self._queue_incoming_betas.get()
            
            # we append em here...
            self.betas.append(betas)

            # simple assignment - or use Kalman or some average..
            self.currentbetas=self.betas[-1]
            #if not self.betas_are_estimated:
            #    self.betas_are_estimated = True

            # self.current_betas = self.betas[-1]
            # we won't do anything with the diff's for now.

                
            # pop if it's getting too much
            #while len(self.betas) > 10:
            #    self.betas.pop(0)

    
        
    def _return_delayed_data(self, data):
        """ because we do CWL regression, we have a delay of some samples - 
            we need therefore to be able to return the data in delayed version
            this returns that 'delayed' data - the d shoulf be below sfuture+spast+1 for this to work...
            
            this returns a shifted version of the data (i.e. backwards in time!)
            we need for handling the input data.. 
            so it also has like a buffer.
        """

        if len(self._buffereddata2) == 0:
            self._buffereddata2 = np.zeros(data.shape)

        self._buffereddata2 = np.concatenate((self._buffereddata2, data))

        if self._buffereddata2.shape[0] > len(self.delayvec) + data.shape[0]:
            self._buffereddata2=self._buffereddata2[-len(self.delayvec)-data.shape[0]:,:]

        
        return self._buffereddata2[-self.zerodelayindex - data.shape[0]:-self.zerodelayindex,:]

        

        
    def _buffer_data(self, data, nsamples):
        """ since we deal potentially with delayed versions of data - we would need to have a small buffer
            which we use to recall stuff from.
            It CAN be that the amount of data coming in, is smaller than the desired size of the buffer
            so solve that here.
            # this will just update the _buffereddata
            
            THIS works in conjunction with _handle_delay_taper_cwl_channel_selection
            To more easily handle the delays there (and select appropriate data)
            
        """
        
        # ipdb.set_trace()
        
        if len(self._buffereddata) == 0:
            self._buffereddata = np.zeros(data.shape)
        while self._buffereddata.shape[0] < nsamples+data.shape[0]:
            self._buffereddata = np.concatenate((np.zeros(data.shape), self._buffereddata))
        
        self._buffereddata = np.concatenate((self._buffereddata, data))
        
        if self._buffereddata.shape[0] + data.shape[0] > nsamples:
            self._buffereddata = self._buffereddata[-nsamples-data.shape[0]:,:]
        
    


    

        
    