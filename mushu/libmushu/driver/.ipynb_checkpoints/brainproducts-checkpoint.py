from __future__ import division

import time
import logging

import numpy as np
import pylsl
import socket
import sys
import time

from libmushu.amplifier import Amplifier


logger = logging.getLogger(__name__)
logger.info('Logger started.')




# it IS possible to HERE do a function definition that'll be 'sub-processed' later on.


# in addition -- it's possible to here also define the functions needed to interface with the
# amplifier, i.e. the python-3 equivalents of the python2 code supplied by BP as a first help.




class BPAmp(Amplifier):


    def __init__(self):

        # call the super...
        super(BPAmp, self).__init__()
        # set some amp-specific values...
        self.backgroundbuffsize=0
        self.expectedreadout=0
        self.experimentnumber=''
        self.sock=None

    """Pseudo Amplifier for Brain Products MR-compatible systems.
    These systems are actually quite quite old, but still somenow the go-to
    for anything related to MR-EEG.


    This amplifier connects to ... well, connecting to maybe is too 'active'.
    It really rather tries to listen to the (directed) network traffic generated
    by the RDA option in the Recorder software. Incidentally, this is the
    exact same method of operation employed by RecView. Which is it's time
    way ahead.

    There should be an ipyNB explaining exactly how to use mushu in combination
    with the Recorder.
    This implementation is ESSENTIAL to allow my work to combine
    BP systems with the mushu - wyrm - pyff way of doing EEG-NF things in
    Python Jupyter Notebooks. Which I think is THE way of handling any NF
    experiment since it'd give experimenter total control, keep things
    simple, and force them to be more closer to the computer and coding practices.


    """

    def configure(self,
                  remotecontrol=True,
                  recorderip='10.100.0.3',
                  pathtoworkspace='C:\\Vision\\Workfiles\\NF_64chEEG.rwksp',
                  experimentnumber='Pre-Run01',
                  subjectid='0001',
                  backgroundbuffsize=100,
                  expectedreadout=0.020,
                  **kwargs):
        """Configure the BP device.

        What this will do is to remotely set all setting properly and start the Recorder
        in Monitoring Mode. You'd have to do impedance check yourself.

        Remote control of the Recorder is done with the Remote Control 1.1. for BrainVision
        Recorder made by Thomas Emmerling in 2009. Ages ago.

        ... aaahhh, I think I could set here an option relating to HOW BIG my
        rolling data buffer should be. But then -- how to save data?

        Here it should be speficied exactly HOW LONG the background ringbuffer will be.
        In addition, you can specify - using some default values -
            1) whether to 'forward' to another UDP socket.
            2) what the (expected) data block should be -- since BP have datablocklength
               of about 20 MSEC. IF a get_data is called and we get >20 MSEC --> place
               notification marker into the EEG (well -- in here). For later inspection.
            3) What the size of the big double-buffer matrix should be. V

            4) What the experiment's workspace should be V

            5) The Experiment's name V

            6) The SUBJECT's name. (hmm.) V


        """

        # solve this sometime later -- behaviour should be to get some kind of popup
        # and a button to say that you're ready.
        if remotecontrol is False:
            raise NotImplementedError()

        # do all pre-actions. See Remote Control v.1.1. document for explanations.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_address = (recorderip, 6700)

        print('connecting to {} port {}'.format(*server_address))
        sock.connect(server_address)

        messages = [b'1' + pathtoworkspace.encode(),
                    b'2' + experimentnumber.encode(),
                    b'3' + subjectid.encode(),
                    b'4',
                    b'M']

        # send messages with 1 sec in between them...
        for message in messages:
            print('sending {!r}'.format(message))
            sock.sendall(message)

            # the Remote Controller doesn't send anything back, so...
            # not check.
            time.sleep(1)

        self.backgroundbuffsize = backgroundbuffsize
        self.expectedreadout = expectedreadout
        self.experimentnumber = experimentnumber
        self.sock = sock

        logger.debug('Remotely Starting BP Amp in Monitoring Mode.')

    def start(self, **kwargs):
        """ This should set things up, and wait until we press 'start' Once

            This should also 'start' the Process to (only) keep track of the Big Matrix.

            IN ADDITION --> it should make you wait - or something -- UNTIL you pressed
             " Start ".
             AND GIVE  A NICE BIG POPUP telling you that you need to press 'record'.

            But --> let's first start checking out the BP documentation(s).

            hmm... OK... so... can I install & use "remote control" ??
            Ah, Nonooo.. It would actually work with Linux, as well.
            So we can just require that this needs to be installed, which is great.
            otherwise.. we just use the button press, of course.

            After checking out the Recorder + Remote Controller for some time, I think

            BP needs to be configured with the workspace, the Experiment, etc etc.
            In principle, it'd be cleaner to dot this down.
        """

        # so.. if this is part of the extra passed arguments --> change it!
        if 'experimentnumber' in kwargs.keys():
            experimentnumber=kwargs['experimentnumber']
            self.sock.sendall(b'2' + experimentnumber.encode())
            time.sleep(1)

        self.sock.sendall(b'4')  # superfluously send '4' again.
        time.sleep(1)
        self.sock.sendall(b'M')  # so when Monitoring Mode is ON --> RDP'll be active.
        time.sleep(1)
        self.sock.sendall(b'D')
        time.sleep(3)
        self.sock.sendall(b'S')

        logger.debug('Starting Recording.')

        # so we CAN start the recording AFTER Monitoring has been initiated.
        # I guess we can close connection, too, right?

        # the file from the Recorder uses completely different start/stop logic as the file from
        # what we try to do here.

        # Here, we'd need to call / initialize our 'container' method.
        # The container method will 'connect' to the server.
        # How will we handle the internal data matrix?
        # check out my ideas. Maybe Decorate Numpy Array will become a to-do.
        # for now, we will proceed with Container.
        # Re-sending the bytecode (exactly) will be the final part of the on-line artifact correction
        # This means we'd need to save it somewhere (in container class)

        # you're already monitoring...



    def stop(self):
        """Close the lsl inlets.

            This should also 'stop' the Process to (only) keep track of the Big Matrix.

            It's not needed to actually do anything anymore as far as we're concerned.


        """

        self.sock.sendall(b'Q')
        time.sleep(1)


        # 'stop' the Underlying Process, too.

        # close the socket (this might close also the Recorder Software)
        self.sock.sendall(b'X')
        self.sock.close()

        logger.debug('Stopping the Recording... going back to Monitoring Mode...')

    def get_data(self):
        """Receive a chunk of data an markers.

        Returns
        -------
        chunk, markers: Markers is time in ms since relative to the
        first sample of that block.

        So... this should, depending on the current time, get ALL samples from the Big DB matrix

        Then --> Perform several sanity checks.
        Check for 1) Missed Blocks (already checked for with the backup Process)
        Check for 2) timing issues (i.e. if it takes TOO long to obtain data)
        Check for 3) boundary issues (i.e. whether there's enough data in Big Matrix to fill all the time,.
        Each of these checks warrants their own little marker.

        Obtain both markers as well as data from the BP system.

        Since this process and the sub-process uses a shared resource (i.e., the Big Matrix)
        Use some kind of locking mechanism to safeguard against a simultaneous read/write.
        NOT sure if it's absolutely necessary. Since in the Container (that we use after all, sigh)
        # maybe we can decorate a numpy array.. though.. , we can implement a get_data method that'll
        # do the housekeeping for us.

        """




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

