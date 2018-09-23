Real-Time Neurofeedback Environment in Python
=============================================
There are two parts:
- rtime, python 3
- stim, python 2

This is the **rtime** part of the NF tools

TODO: (later) this information






New Version of Pyff - Use for Neurofeedback
===========================================

The following documentation I recommend to read through before you try to use any of this, since it explains many of the ideas and concepts that you need if for setting up your real-time EEG NF system in Python.

- The Features of Pyff, or Why Is It Important?
- Installation
- Psychopy Integration
- Starting, Stopping, Sending from Another Machine
- A Note on how Pyff works, that's not documented anywhere




The Features of Pyff, or Why Is It Important?
---------------------------------------------

The idea is to use free, open-sourced, software-license free programs to perform Neurofeedback experiments. In the python world, only psychopy (optimizing for stimulus generation, timing in sofar possible, logging) and pyff (solving the problem of real-time update of stimuli and experiments depending on experimental data) are available. They also offer core functionality that supplements each other.

| Pyff| Psychopy  |
|----------|------|
| start/stop experiments | visual stimuli |
| remote control from another computer | auditory stimuli |
| send data to experiment in RT | keyboard/button interface |
| | logging to logfile |
| |use extra/dual screens |


Together, they provide the ideal solution for stimulus presentation and real-time adaptation of visual and auditory stimuli. Of course, Pyglet (or even Pygame), which are more basic libraries, also support these things, but Psychopy here makes your life a LOT easier.

A typical workflow exists of, when it comes to stimulus computer:

1. Start up the lab's specific Stimulus computer
2. Double-click the icon to start up FeedbackController
3. WHEN DOING RT: Use your real-time fMRI computer to send signals to the Stimulus computer via TCP/IP to start/stop any (pre-built) experiment and send signals to the (running) NF Experiment
4. NOT DOING RT: Use the Stimulus or the RT Computer computer to start/stop a non-RT Experiment



Installation
------------

The Pyff + Psychopy should not be installed 'like that' on your system. It is highly recommended to use a conda environment for that, otherwise chances are high things won't run. This is not just true for this Python package, but true for ALL Python packages in which some development is going on. The main reason is that Pyff is a rather old package, so won't work with the most up-to-date version of pyqt. Also it is coded in Python = 2.7, and it'd take serious effort to try to convert to python = 3.5. In addition, Psychopy is (as of yet) coded in python 2.7. The use of a conda environment guarantees that the packages will run as intended on any system. In addition, it's rather easy to make a shortcut to start things up.

The way to install this stimulus presentation is:

1. Install Git
2. Install Conda
3. Grab Repository of Pyff (using Git)
4. Use Conda to set up the correct environment
5. Make the shortcut to easily quick-start the FeedbackController

```console
$ git clone github.com/jnvandermeer/pyff
$ cd pyff
```

Then, once inside, use conda to install the environment, including psychopy!

```console
$ conda install environment.xml
```


Psychopy Integration
--------------------

You cannot use the Builder - you must use/work with the Coder! Do not worry, for everything that's more difficult than the most basic stuff, you need to code in Presentation, too. Same for OpenVibe or for BCI2000, or Psych Toolbox.

The price you pay for doing things in a scapable manner is that you'd have to work with python.

Some words on Psychopy: Psychopy has been integrated as a FeedbackBase class, located in /src/FeedbackBase, which does all of the initialization stuff. Psychopy uses (and should work) with Pyglet, so we can solve the issue of multiple screens.




Starting, Stopping, Sending from Another Machine
------------------------------------------------







A Note on how Pyff works, that's not documented anywhere
--------------------------------------------------------











How to use the FeedbackController
=================================

Enter the src directory and start FeedbackController:

    cd src
    python FeedbackController.py


If you made changes to the gui.ui or just checked out the tree from SVN, you
may have to rebuild the gui.py:

    cd src
    make

How to use the parallel port under linux
========================================

```bash
$ sudo modprobe -r lp
$ sudo chmod 666 /dev/parport0
```

Citing Us
=========

If you use Pyff for anything that results in a publication, We humbly ask you to
cite us:

```bibtex
@ARTICLE{venthur2010,
    author={Venthur, Bastian  and  Scholler, Simon  and  Williamson, John  and  Dähne, Sven  and  Treder, Matthias S  and  Kramarek, Maria T  and  Müller, Klaus-Robert  and  Blankertz, Benjamin},
    title={Pyff---A Pythonic Framework for Feedback Applications and Stimulus Presentation in Neuroscience},
    journal={Frontiers in Neuroinformatics},
    volume={4},
    year={2010},
    number={100},
    url={http://www.frontiersin.org/neuroinformatics/10.3389/fninf.2010.00100/abstract},
    doi={10.3389/fninf.2010.00100},
    issn={1662-5196},
}
```

