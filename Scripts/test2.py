import numpy as np
import matplotlib.pyplot as plt
from src.ML_funcs import return_artifical_data
from src.utils import DataUtils
from src.traces import Traces
import pandas as pd
from tsfresh import extract_features

# data, labels = return_artifical_data(500,1.7,5)
#
#
# y = data[0]
# x = np.arange(len(y))
#
# df = pd.DataFrame({'value':y})
# df['time'] = x
# df['id'] = 0
#
# if __name__ =='__main__':
#     features = extract_features(df,column_id='id', column_sort='time')
#     var = features.values

import numpy as np
import pandas as pd
from tsfresh import extract_features

def extract_features_for_time_series(y, idx):
    x = np.arange(len(y))

    # Create a DataFrame for the time series data
    df = pd.DataFrame({'value': y})
    df['time'] = x
    df['id'] = idx  # Use the index as the identifier

    # Extract features using tsfresh
    features_df = extract_features(df, column_id='id', column_sort='time')

    # Convert the features DataFrame to a numpy array and return it
    features_array = features_df.values
    return features_array

def main():
    # Your data and labels, replace with actual values
    data, labels = return_artifical_data(500, 1.7, 5)

    # Initialize an empty list to store feature arrays
    all_features = []

    # Iterate through each time series
    for idx, y in enumerate(data):
        print(idx, idx/len(data)*100 )
        features_array = extract_features_for_time_series(y, idx)
        all_features.append(features_array)

    # Convert the list of feature arrays to a list of numpy arrays
    all_features_arrays = all_features

    print("Features for all time series as a list of numpy arrays:")
    print(all_features_arrays)

    # Save the list of numpy arrays
    np.save('all_features_arrays.npy', all_features_arrays)

if __name__ == '__main__':
    main()



