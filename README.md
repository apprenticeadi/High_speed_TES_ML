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
2. Make changes to the code and commit. 
```commandline
git add --all
git commit -m 'Your commit message' 
```

3. Push the new local branch to the remote repository with tracking with the --set-upstream flag (-u for short):
```commandline
git push -u origin new_branch_name
```

4. After the first push, subsequent changes can be commited and pushed more easily. 
```commandline
git add --all
git commit -m 'Your commit message' 
git push 
```

### Required packages 
The required packages can be installed by creating a conda virtual environment from the environment.yml file. In your 
anaconda prompt or command line, type the following to create a conda environment named 'tes_ml': 
```commandline
conda env create -f environment.yml 
```

## About the code 

## Example code 




