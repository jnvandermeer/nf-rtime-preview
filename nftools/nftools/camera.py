import easygui
import threading
from telnetlib import Telnet

class DoCamera(threading.Thread):
    ''' to kind-of automatize the camera action...
    '''
    
    def __init__(self, ip='20.100.0.201', port=6802):
        super(DoCamera, self).__init__()
        self._state='RECORDING'
        self.current_counter = 1
        self.ip = ip
        self.port = port
        self.txtoptions = ('Pause Recording', 'Quit')
    
 
    def run(self):
        
        runCommand = False
        prev_state = 'initial'
        
        stoploop = False
        while not stoploop:
            
                        
            if self._state != prev_state:
                runCommand = True
            else:
                runCommand = False
            
            prev_state = self._state
            
            
            if runCommand:
                if self._state == 'RECORDING' or self._state == 'Continue RECORDING':
                    tnstr = '1|on|100|Measurement|Participant Recording with the Axis Camera|Measuring EEG'
                    self.txtoptions = ('Pause Recording','Quit')
                    
                if self._state == 'Pause Recording':
                    tnstr = '1|cancel|10|Measurement|Participant Recording with the Axis Camera|'
                    self.txtoptions = ('Continue RECORDING','Quit')


                with Telnet(self.ip, self.port) as tn:
                    tn.write(tnstr.encode('ascii')+b'\n')


            current_status_text = self._state
            if current_status_text == 'Continue RECORDING':
                current_status_text = 'RECORDING'

            self._state = easygui.buttonbox('Camera is currently on: %s' % current_status_text,'Camera Control', self.txtoptions)

            if self._state == 'Quit':
                tnstr = '1|cancel|10|Measurement|Participant Recording with the Axis Camera|'
                stoploop = True
                
                with Telnet(self.ip, self.port) as tn:
                    tn.write(tnstr.encode('ascii')+b'\n')

                
            

    def get_state(self):
        if self._state == 'Stop':
            print('loop is stopped!')
        return self._state
    



