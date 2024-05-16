# Software package for TES discrimination

This package provides the code for discriminating TES voltage signals. The main methods are an inner product classifier 
and a machine learning tabular classifier (e.g. Random Forest). 

### Installation
**Branch protection is not enabled for this repository. PLEASE DO NOT PUSH OR MERGE INTO THE MAIN BRANCH.**

This repository will be moved into Imperial College Github organisation soon to enable better team collaboration. 

The package can be installed by cloning the repository: 
```sh
git clone https://github.com/apprenticeadi/TES_python.git
```

After installation, **please create a new branch** by following the steps below. 
1. Create a new local branch from existing branch (e.g. main)
```commandline
git checkout -b new_branch_name existing_branch_name
```
2. 

The required packages can be installed by creating a conda virtual environment from the environment.yml file. In your 
anaconda prompt or command line, type the following to create a conda environment named 'tes_ml': 
```commandline
conda env create -f environment.yml 
```


