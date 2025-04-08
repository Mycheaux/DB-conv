# Bi-directional DB-converter 
This GitHub repository contains the tool used in the paper titled `Self-supervised generative AI enables conversion of two non-overlapping cohorts`. It is a self-supervised deep learning architecture leveraging category theory designed to convert data from different cohorts with different data structures into each other. This takes input of the starting (DB1) and target data(DB2) in train, dev, and test splits, and outputs a trained model which contains weights and biases for both the forward DB-converter ($m$) and backward DB-converter ($i$). Based on the provided test set of DB1 and DB2, it also outputs $m(DB_1)$ as Converted-DB1 (which is in $DB_2$ scheme), $i(m(DB_1)$ as Reconverted-DB1, $i(DB_2)$ as Converted-DB2(which is in $DB_1$ scheme) and $m(i(DB_2)$ as reconverted DB_2. 
![Alt Text](GithubFig.png)

# How to use DB-converter
We deploy this app in 3 possible ways. 1. This GitHub (one need to set up own environment) 2. Google collab (one doesn't need to set up the environment) 3. Docker image version (one doesn't need to set up the environment). 



# Solution 1: Directly run this Github repo

## Environment Set up:
We currently provide both CPU and GPU (NVIDIA) support. The app is tested on a Mac M1 CPU environment.
It should run fine in a Linux environment without necessary changes. If you are on Windows and find errors due to the path due to `/` vs \' you may try the following fix:You can create a batch script that automatically translates paths with / into \ before passing them to tools that require backslashes. For example:
```
@echo off

set "input_path=%1"

set "converted_path=%input_path:/=\%"

echo %converted_path%
```
If you prefer not to deal with Windows path quirks, you can run your repository in a Unix-like environment such as: Windows Subsystem for Linux (WSL),Git Bash.

### Python Libraries:
We assume we already have anaconda or miniconda if not check here how to get one. https://www.anaconda.com/download or https://www.anaconda.com/docs/getting-started/miniconda/main

1. Best, most general way:
```
     conda create --name myenv python=3.12 -y
     conda activate myenv
     conda install pip -y
     conda install numpy scipy pandas -y
```
   Now, install either cpu or the GPU version of PyTorch
   
   CPU version:
   
     conda install pytorch=2.3 torchvision torchaudio cpuonly -c pytorch  # CPU-only
   
   GPU version:
   
     conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia  # GPU-only
     
   Now install lightening API and weights and biases
```
conda install lightning -c conda-forge
conda install wandb -c conda-forge
```
 
3. In general for any CPU environment, you should first create a fresh conda environment using
```
conda create --name myenv python=3.12 -y
conda activate myenv
conda install pip -y
```
   and install all packages
   
     pip install -r requirements.txt
     
   This method breaks when pip can't find the right version, try the conda alternative as suggested above.
4. In general for any devices with availavle CUDA-supported NVDIA GPU, environment you should first create a fresh conda environment using
```
conda create --name myenv python=3.12 -y
conda activate myenv
conda install pip -y
pip install -r requirements.txt
```
   This method breaks when pip can't find the right version, try the conda alternative as suggested above.
5. If you are using Mac M1 2021 try this if you encounter dependency problems in the general way. 

     conda env create -f m1cpu_environment.yml
  
then from activated environment

     pip freeze > m1cpu_requirements.txt  # From within activated environment
  
  This method breaks when pip can't find the right version, try the conda alternative as suggested above.

### Weights and Biases API:
You require a weights and bias account to monitor your Model training. Make a free account here : https://wandb.ai/site , and find your API key in your settings after logging in. When the app starts, it will ask for your API key to log in, and the App with create loss function plots during training.






### It uses two NN architectures trained in tandem

To run the app, you need a weights and bias account; it will ask for the API.

Currently, the `requirement.txt` is ignored; you can reactivate it from `main.py`.

Please install the correct version of pytorch depending on the availability of CUDA devices/ change the `requirement.txt` accordingly.

You can use existing folders for input data and output model, or you can use a different location, change the path in `config/data_path.yaml`

Most hyperparameters that are of interest to be tuned for most tasks can be found in `config/congif.yaml` and `config/architecture.yaml`

Other hyperparameters are in `config/advanced_config.yaml`

The user can define their own mapper (forward DB- converter) and inverter (backward DB- converter) by replacing the model.py or changing 2nd line of `db-converter.py` file

