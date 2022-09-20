---
title: Setup
---

:::::::::::::::::: prereq
Before attempting today's lessons, you will need to have installed the following:

- Python 3.9
- conda
- pip3

:::::::::::::::::: 

If you have previously attended the first 2 days of the workshop, it is most likely that the above installations and prerequisites would have been met. Now the next step is to create a new environment, where we will install all the modules, tools and packages required for this particular lesson.

## Setting up virtual environment
<p style='text-align: justify;'>
In Python, the use of virtual environments allows you to avoid installing Python packages globally, which could disrupt system tools, or other projects.  Each virtual environment has its own Python binary (which matches the version of the binary that was used to create this environment), and can have its own independent set of installed Python packages within its site directories.
</p>

1. A virtual environment can be created by executing the following command in your Terminal (Mac OS and Unix), or on the command line prompt (Windows):

```
conda create --prefix ./envL2D python=3.9 pip
```

By running this command a new environment will be installed within your home directory.

2. The environment can be activated as:

```
conda activate ./envL2D 
```
If you have an Apple Mac running Apple Silicon (to include any of M1 and M2 series chips), skip to step 4, otherwise follow step 3, and then skip straight to step 9 (bypassing steps 4-8).

3. The packages required for this lesson are in the file [requirements.txt](data/requirements.txt). In order to install them in your environment, write this command on your terminal. Please note, that you will need to explicitly provide the filepath for your requirements.txt file, or have it located within your current working directory.

```
 envL2D/bin/pip3 install -r requirements.txt
```

4. If you are running a machine with Apple Silicon (M1x or M2 chips), then you may need to install tensorflow separately, which can be done, as follows:
   
   Install TensorFlow dependencies from the Apple Conda channel:

```
conda install -c apple tensorflow-deps
```

5.  Install base TensorFlow (Apple's fork of TensorFlow is called tensorflow-macos):

```
python -m pip install tensorflow-macos
```

6.  Install Apple's tensorflow-metal to leverage Apple Metal (Apple's GPU framework) for M1x and M2 GPU acceleration:

```
python -m pip install tensorflow-metal
```

7. (Optional, but recommended) Install TensorFlow Datasets to run benchmarks included in this repo:

This environment kernel needs to be added to your Jupyter notebook. This can be done as:

```
python -m pip install tensorflow-datasets
```

8. Install all the remaining required packages from the file [requirements2.txt](data/requirements.txt). This file does not have tensorflow packages as they have already been installed in steps 4-7 for users running Apple Silicon.


9. At this point, everything will have been installed in our new environment. This is now the time to add this environment in your Jupyter Notebook, which can be done as follows:

```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=envL2D
```
<p style='text-align: justify;'>
After running these 2 commands, you will be able to select your virtual environment from the `Kernel` tab of your Jupyter notebook. More information can be accessed at this [link](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084).
</p>

## Dataset
Dataset for this lesson includes:
[human_ht29_colon_cancer_2_images](data/human_ht29_colon_cancer_2_images.zip)

