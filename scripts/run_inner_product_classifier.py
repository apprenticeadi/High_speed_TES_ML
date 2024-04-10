import numpy as np
import matplotlib.pyplot as plt

from tes_resolver.classifier.inner_product import InnerProductClassifier
from tes_resolver.traces import Traces
from src.data_utils import DataReader
import tes_resolver.config as config

'''Run the inner product classifier'''

dataReader = DataReader('RawData')

data_group = 'raw_8'
rep_rate = 100

data_raw = dataReader.read_raw_data(data_group, rep_rate)
calTraces = Traces(rep_rate, data_raw, parse_data=True, trigger=0)

ipClassifier = InnerProductClassifier(multiplier=1., num_bins=1000)
ipClassifier.train(calTraces)

labels = ipClassifier.predict(calTraces, update=True)

pns, freq = calTraces.pn_distribution(normalised=True)

mean_photon_number = np.sum(pns * freq)
print(f'mean photon number is {mean_photon_number}')