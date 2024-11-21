# Software package for TES discrimination

This package provides the code for discriminating TES voltage signals. The main methods are an inner product classifier, 
a supervsied machine learning classifier (e.g. Random Forest), and an unsupervised clustering classifier. The methods are based on [insert paper]

### Installation
Users interested in using the code are welcome to fork this repository. 

### Required packages 
The code is written in Python 3.10 and requires the following packages:
```text
-numpy==1.26.4
-matplotlib=3.8.4
-scipy=1.13.1
-pandas-2.2.2
-scikit-learn==1.4.2
-py-xgboost=2.1.1
-hdbscan=0.8.38
-tensorflow==2.10.1
```

An easy way to install the required packages is to create a conda environment from the environment2.yml file, which 
specifies an anaconda environment. In your anaconda prompt or command line, type the following to create the 
environment, 'tes_env': 
```commandline
conda env create -f environment2.yml 
```

## Loading data
Once installed and the environment is activated, place the TES voltage data in the 'Data' folder. The example data used in 
our paper can be provided upon request to the authors, the format of which is: 
```commandline
├───Data
│   ├───Tomography_data_2024_04
│       ├───power_0
│           ├───100kHz.txt
│           ├───200kHz.txt
│           ├ ...
│           ├───1000kHz.txt
│       ├───power_1
│       ├...
│   ├───squeezed states 2024_07_17
│       ├───2024-07-17-1954_100kHz_2nmPump_1570nmBPF_112uWpump_Raw_Traces_Chan[1]_1.txt
│       ├───2024-07-17-1954_100kHz_2nmPump_1570nmBPF_112uWpump_Raw_Traces_Chan[2]_1.txt
│       ├ ...
│       ├───2024-07-17-2010_800kHz_2nmPump_1570nmBPF_900uWpump_Raw_Traces_Chan[1]_1.txt
│       ├───2024-07-17-2010_800kHz_2nmPump_1570nmBPF_900uWpump_Raw_Traces_Chan[2]_1.txt
│       ├ ...
```

When stored in the correct format, the example data can be loaded with DataReader from utils. Example code is shown 
'scripts\read_data.py'. Alternatively, try the code snippet below: 
```python
from utils.data_reader import DataReader, RuquReader
from tes_resolver import Traces

rep_rate = 800
# read coherent state data
dataReader1 = DataReader('Data/Tomography_data_2024_04')
data_group = 'power_6'
data_coh = dataReader1.read_raw_data(data_group, rep_rate)
cohTraces = Traces(rep_rate, data=data_coh, parse_data=True)

# read squeezed state data
dataReader2 = RuquReader(r'Data/squeezed states 2024_07_17')
data_keywords = [f'{rep_rate}kHz', '2024-07-17-2010_', '2nmPump', '1570nmBPF', 'Chan[1]']
data_sq = dataReader2.read_raw_data(*data_keywords, concatenate=True)
sqTraces = Traces(rep_rate, data=data_sq, parse_data=True)

```

## Processing data
Example code using inner-product and machine learning classifiers can be found in 'scripts\ip_example.py', 
'scripts\ml_example.py', and 'scripts\hdbscan_example.py'.

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
