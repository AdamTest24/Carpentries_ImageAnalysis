---
title: Setup
---

:::::::::::::::::: prereq
- Python 3.9
- conda
- pip3
:::::::::::::::::: 

With first 2 days at the workshop, it is most likely that the above prerequisites would have been met. Now the next step is to create an environment where we could install all the tools/packages required for this particular lesson.

## Setting up virtual environment
<p style='text-align: justify;'>
In Python, the use of virtual environments allows you to avoid installing Python packages globally which could break system tools or other projects.  Each virtual environment has its own Python binary (which matches the version of the binary that was used to create this environment) and can have its own independent set of installed Python packages in its site directories.
</p>

1. A virtual environment can be created by executing the following command in your Terminal (Mac OS and Unix) or at the command line prompt (Windows):

```
conda create --prefix ./envL2D python=3.9  
```

By running this command a new environment will be installed at your home directory.

2. The environment can be activated as:

```
conda activate ./envL2D 
```
If you have Mac M1 chip, skip to step 4, otherwise follow step 3 and then skip to step 10.

3. The packages required for this lesson are in the file [requirements.txt](data/requirements.txt). In order to install them in your environment, write this command on your terminal:

```
 pip3 install -r requirements.txt
```
Note: You need to download this file in your current working directory to use this.

4. In case you have Mac M1 chip then you may need to install tensor flow separately which can be done as follows:

5. Install TensorFlow dependencies from Apple Conda channel.

```
conda install -c apple tensorflow-deps
```
6.  Install base TensorFlow (Apple's fork of TensorFlow is called tensorflow-macos).

```
python -m pip install tensorflow-macos
```
7.  Install Apple's tensorflow-metal to leverage Apple Metal (Apple's GPU framework) for M1, M1 Pro, M1 Max, M1 Ultra, M2 GPU acceleration.

```
python -m pip install tensorflow-metal
```
8. (Optional) Install TensorFlow Datasets to run benchmarks included in this repo.

This environment kernel needs to be added to your Jupyter notebook. This can be done as:
```
python -m pip install tensorflow-datasets
```

9. Install the rest of packages from the file [requirements2.txt](data/requirements.txt). This file does not have tensor flow packages as they have already been installed on Mac M1.


10. Now everything has been installed in our environment. Now this is the time to add this environment in your jupyter notebook which can be done as follows:

```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=envL2D
```
<p style='text-align: justify;'>
After running these 2 commands, you will be able to select your virtual environment from the `Kernel` tab of your Jupyter notebook. More information can be accessed at this [link](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084).
</p>

## Dataset
Dataset for this lesson includes:


