import numpy as np
import matplotlib.pyplot as plt
from src.ML_funcs import return_artifical_data
from src.utils import DataUtils
from src.traces import Traces
import pandas as pd
from tsfresh import extract_features
data, labels = return_artifical_data(500,1.7,5)


y = data[0]
x = np.arange(len(y))

df = pd.DataFrame({'value':y})
df['time'] = x
df['id'] = 0
var = 0
if __name__ =='__main__':
    features = extract_features(df,column_id='id', column_sort='time')
    var = features


