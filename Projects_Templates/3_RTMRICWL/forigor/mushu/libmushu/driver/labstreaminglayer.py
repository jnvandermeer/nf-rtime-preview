

import time
import logging

import numpy as np
import pylsl
import ipdb

from libmushu.amplifier import Amplifier


logger = logging.getLogger(__name__)
logger.info('Logger started.')


class LSLAmp(Amplifier):
    """Pseudo Amplifier for lab streaming layer (lsl).

    This amplifier connects to an arbitrary EEG device that is
    publishing its data via lsl to the network. With this class you can
    use this amplifier like a normal `mushu` amplifier.

    https://code.google.com/p/labstreaminglayer/

    Examples
    --------

    >>> amp = libmushu.get_amp('lslamp')
    >>> amp.configure()
    >>> amp.start()
    >>> while True:
    ...     data, marker = amp.get_data()
    ...     # do something with data and/or break the loop
    >>> amp.stop()

    """

    def configure(self, **kwargs):
        """Configure the lsl device.

        This method looks for open lsl streams and picks the first `EEG`
        and `Markers` streams and opens lsl inlets for them.

        Note that lsl amplifiers cannot be configured via lsl, as the
        protocol was not designed for that. You can only connect (i.e.
        subscribe) to devices that connected (publishing) via the lsl
        protocol.

        """
        
        self.markers_list=[]
        # lsl defined
        #self.max_samples = 4096
        self.max_samples = 1024
        # open EEG stream
        logger.debug('Opening EEG stream...')
        streams = pylsl.resolve_stream('type', 'EEG')
        if len(streams) > 1:
            logger.warning('Number of EEG streams is > 0, picking the first one.')
        self.lsl_inlet = pylsl.StreamInlet(streams[0], max_buflen=30)
        # open marker stream
        logger.debug('Opening Marker stream...')
        # TODO: should add a timeout here in case there is no marker
        # stream
        streams = pylsl.resolve_stream('type', 'Markers')
        if len(streams) > 1:
            logger.warning('Number of Marker streams is > 0, picking the first one.')
        self.lsl_marker_inlet = pylsl.StreamInlet(streams[0], max_buflen=30)
        info = self.lsl_inlet.info()
        self.n_channels = info.channel_count()
        self.channels = ['Ch %i' % i for i in range(self.n_channels)]
        self.fs = info.nominal_srate()
        logger.debug('Initializing time correction...')
        
        
        self.do_time_correction = True
        try:
            self.lsl_marker_inlet.time_correction(timeout=2.0)
        except Exception:
            self.do_time_correction = False
            logger.debug('Timeout whe tying marker_inlet time_correction')
            
        try:
            self.lsl_inlet.time_correction(timeout=2.0)
        except Exception:
            self.do_time_correction = False
            logger.debug('Timeout whe tying marker_inlet time_correction')
            
            
        logger.debug('Configuration done.')

    def start(self):
        """Open the lsl inlets.
        
        """
        logger.debug('Opening lsl streams.')
        self.lsl_inlet.open_stream()
        self.lsl_marker_inlet.open_stream()

    def stop(self):
        """Close the lsl inlets.

        """
        logger.debug('Closing lsl streams.')
        self.lsl_inlet.close_stream()
        self.lsl_marker_inlet.close_stream()

    def get_data(self):
        """Receive a chunk of data an markers.

        Returns
        -------
        chunk, markers: Markers is time in ms since relative to the
        first sample of that block.

        """
        if self.do_time_correction:
            # print('doing time correction')
            tc_m = self.lsl_marker_inlet.time_correction()
            tc_s = self.lsl_inlet.time_correction()
        else:
            tc_m = 0.0
            tc_s = 0.0

        markers, m_timestamps = self.lsl_marker_inlet.pull_chunk(timeout=0.0, max_samples=self.max_samples)
        # flatten the output of the lsl markers, which has the form
        # [[m1], [m2]], and convert to string
        markers = [str(i) for sublist in markers for i in sublist]
        
        if len(markers)>0:
            for itemi, item in enumerate(markers):
                self.markers_list.append([markers[itemi], m_timestamps[itemi]])

        # block until we actually have data
        #samples, timestamps = self.lsl_inlet.pull_chunk(timeout=pylsl.FOREVER, max_samples=self.max_samples)
        samples, timestamps = self.lsl_inlet.pull_chunk(timeout=0.0, max_samples=self.max_samples)
        samples = np.array(samples).reshape(-1, self.n_channels)
        
        
        #if len(m_timestamps) > 0 and len(timestamps) > 0:
        #    ipdb.set_trace()



        if len(timestamps) > 0:
            markers=[]
            m_timestamps=[]
            topop=[]
            for mi, item in enumerate(self.markers_list):
                mtype=item[0]
                mts=item[1]
                if mts in timestamps:
                    markers.append(mtype)
                    m_timestamps.append(mts)
                    topop.append(mi)
    
            # remove em if found..
            if len(topop) > 0:
                for popi in topop:
                    self.markers_list.pop(popi)

        else:
            markers=[]
            m_timestamps=[]

        #if len(timestamps) > 0:
            # see if we can find markers that match the data:
        #    for ts in se

        
        if len(timestamps) > 0:

            t0 = timestamps[0] + tc_s
            m_timestamps = [(i + tc_m - t0) * 1000 for i in m_timestamps]

        # so we need to do something here. Put the marker information into a lst
        # then figure out what our timestamps are
        # then if timestamp of DATA -- matches timestamp of MARKER
        # we can send out the marker belonging to that thing, according to recipe below
        
        
        
        
        #if len(timestamps) != len(m_timestamps):
        #    ipdb.set_trace()

        return samples, list(zip(m_timestamps, markers))

    def get_channels(self):
        """Get channel names.

        """
        return self.channels

    def get_sampling_frequency(self):
        """Get the sampling frequency of the lsl device.

        """
        return self.fs

    @staticmethod
    def is_available():
        """Check if an lsl stream is available on the network.

        Returns
        -------

        ok : Boolean
            True if there is a lsl stream, False otherwise

        """
        # Return True only if at least one lsl stream can be found on
        # the network
        if pylsl.resolve_streams():
            return True
        return False

