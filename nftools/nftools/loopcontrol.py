import easygui
import threading
class LoopState(threading.Thread):

    def __init__(self):
        super(LoopState, self).__init__()
        self._state='initial'
    
    def run(self):
        while self._state != 'Stop':
            self._state = easygui.buttonbox('Loop should:','Control over the NF While Loop', ('Pause','Stop'))

    def get_state(self):
        if self._state == 'Stop':
            print('loop is stopped!')
        return self._state
