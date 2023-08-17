import numpy as np
import matplotlib.pyplot as plt
from src.ML_funcs import return_artifical_data
from src.utils import DataUtils
from src.traces import Traces


data100 = DataUtils.read_raw_data_new(100,5)
trace100 = Traces(100,data100,1.8)
art_500 = trace100.generate_high_freq_data(500)


trace500 = Traces(500,art_500,1.8)
av1, er1, er2 = trace500.average_trace(plot = False)

actual_data = DataUtils.read_high_freq_data(500,5,new=True)
ac_trace = Traces(500,actual_data,1.8)
av2, err3,err4 = ac_trace.average_trace()

plt.plot(av1, label = 'artificial')
plt.plot(av2, label = 'real')

plt.show()

