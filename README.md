Real-Time Neurofeedback Environment in Python
=============================================

This is the **rtime** part of the NF tools


A typical workflow exists of, when it comes to the real-time computer:

```
conda activate rt
cd nf-stim
jupyter lab
```

At this stage you should load in one of the ipython notebooks, which should explain the flow and purpose of each experiment. See the Projects_Templates folder for several examples (work in progress!)

The notebook and its structure is used for (iteratively)
- importing all the libraries
- starting/stopping the stimulation (experiment) on the experiment computer running the nf-stim environment
- starting/stopping the feedback loop
- defining the real-time analysis workflow
- starting/stopping the camera (if you have one)
- visualizing the real-time analysis
- changing parameters of the real-time analysis during the experiment (for example the threshold or scaling)
- documenting the purpose and flow of the experiment with markdown
- saving data
- doubling back when something needs to be re-done (happens a lot!)




Installation
============

The way to install this stimulus presentation is:

1. Install Git
2. Install Conda
3. Clone the nf-rtime environment
4. Use Conda to set up the correct environment `conda env create -f environment.yml` (there are different yml files for windows and linux!)

