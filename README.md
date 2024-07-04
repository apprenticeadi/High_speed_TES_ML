# Software package for TES discrimination

This package provides the code for discriminating TES voltage signals. The main methods are an inner product classifier 
and a machine learning tabular classifier (e.g. Random Forest). The methods are based on [insert paper]

### Installation
**Branch protection is enabled for the main branch. If you plan to make changes, please create a new branch.**

The package can be installed by cloning the repository: 
```sh
git clone https://github.com/apprenticeadi/TES_python.git
```

After installation, you can create a new branch:
```commandline
git checkout -b new_branch_name existing_branch_name
```
Once you have made changes to the code and wish to push them to the remote repository, follow the commands below:
```commandline
git add --all
git commit -m 'Your commit message' 
git push -u origin new_branch_name
```
This will create a new tracking branch at remote. After the initial push, subsequent changes can be committed and pushed more easily. 
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

## Loading data
Once installed and the environment is activated, place the TES voltage data in the 'Data' folder. The data should be stored
in txt files with delimiter ',' and the repetition rate should be in the file name. An example of the data format is shown below: 
```commandline
├───Data
│   ├───Tomography_data_2024_04
│       ├───power_0
│           ├───100kHz.txt
│           ├───200kHz.txt
│           ├───300kHz.txt
│           ├───400kHz.txt
│           ├───500kHz.txt
│           ├───600kHz.txt
│           ├───700kHz.txt
│           ├───800kHz.txt
│           ├───900kHz.txt
│           ├───1000kHz.txt
```

When stored in the correct format, the data can be loaded with DataReader from utils. Example code is shown below: 
```python
from utils import DataReader
dataReader = DataReader('Data/Tomography_data_2024_04')
data100 = dataReader.read_raw_data('power_0', rep_rate=100)  # load 100kHz data
```
Alternatively, the data can be loaded manually by
```python
import numpy as np 
data100 = np.loadtxt(filename, delimiter=',', unpack=True).T
```
The numpy array is transposed so that the voltage traces are stored in rows, instead of columns. This can be changed 
depending on how the TES data is collected.

## Processing data
Example code using inner-product and machine learning classifiers can be found in 'scripts\ip_example\' and 'scripts\ml_example\' respectively.

Alternatively, try the code snippet below: 
```python
from tes_resolver import Traces, generate_training_traces
from tes_resolver.classifier import InnerProductClassifier, TabularClassifier
from utils import DataReader
# load data 
dataReader = DataReader('Data/Tomography_data_2024_04')
data100 = dataReader.read_raw_data('power_0', rep_rate=100)
data500 = dataReader.read_raw_data('power_0', rep_rate=500)

# create Traces object with unknown labels, sampling rate 50MHz, and parse data such that each row of the data is a voltage trace. 
calTraces = Traces(rep_rate=100, data=data100, labels=None, sampling_rate=5e4, parse_data=True)  # calibration traces
highTraces = Traces(rep_rate=500, data=data500, labels=None, sampling_rate=5e4, parse_data=True)  # high rep-rate traces

# photon number assignment to calibration traces with inner product method
ipClassifier = InnerProductClassifier()
ipClassifier.train(calTraces)  # use the average trace of calTraces as reference trace in inner product
ipClassifier.predict(calTraces, update=True)  # predict the photon number and update the labels attributes of calTraces

# overlap calibration traces to higher rep rate
trainingTraces  = generate_training_traces(calTraces, target_rep_rate=500)

# use random forest classifier to predict higher rep rate data 
rfClassifier = TabularClassifier('RF')
rfClassifier.train(trainingTraces)
rfClassifier.predict(highTraces, update=True)
```
## Citing
We would appreciate citations to the following paper: 
