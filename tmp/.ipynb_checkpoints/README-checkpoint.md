# EEG Neurofeedback with Python


Welcome this is a test...





## Methods in Amplifier class that need to be implemented

These are the methods which should be used by users wishing to interface with the amp

- configure

- start

- stop

- get_data

- get_channels

- get_sampling_frequency

- is_available

## Workflow

### On Starting the Amp

Amplifier should be started separately for the Recorder Computer and the RT Interface Computer (the computer where RT - CWL EEG FMRI artifact correction takes place, or where Wyrm does its magic). I (don't think) it's possible that the amplifier can be started remotely with any API interface such as is the case with EGI and NeuroOne (and maybe others). So the way to go about it is to assume that amps can be started remotely, and work like that, and in the (special) case for BP Recorder, we add in the requirement that users have to press the 'Start' button separately.


