import numpy as np
import matplotlib.pyplot as plt

from tes_resolver.ml_funcs import generate_training_traces
from tes_resolver.traces import Traces

from src.data_reader import DataReader

dataReader = DataReader('RawData')

# overlap the 100kHz data, and compare the average trace with the actual high rep_rate data. How big is the vertical shift?

# maybe also try plotting the raw traces and see how they cluster (training data vs actual data)

