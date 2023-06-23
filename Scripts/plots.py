import numpy as np
import matplotlib.pyplot as plt
from src.utils import DataUtils, TraceUtils
from src.traces import Traces
from src.ML_funcs import ML
import time
import numpy as np
from numpy.polynomial import polynomial
import pandas as pd
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib as mpl


frequency = 900
actual_data = DataUtils.read_high_freq_data(frequency)
targetTraces = Traces(frequency=frequency, data=actual_data)
offset_target, _ = targetTraces.subtract_offset()
actual_data = actual_data - offset_target
fig = plt.figure(figsize=(20 / 2.54, 20 / 2.54))
sins = pd.DataFrame(actual_data, )

ax1 = plt.subplot2grid((3,10), (0,0), colspan=10)
ax2 = plt.subplot2grid((3,10), (1,0), colspan=10)
ax3 = plt.subplot2grid((3,10), (2,0), colspan=9)
ax4 = plt.subplot2grid((3,10), (2,9))
print('done1')
# plot line data
sins.T.plot(ax=ax1, color='r',linewidth=.001)
ax1.legend_.remove()
ax1.set_xlim(0, len(actual_data[0]))
ax1.set_title('900 kHz')
print('done2')
# try boxplot
sins.plot.box(ax=ax2, showfliers=False)
xticks = ax2.xaxis.get_major_ticks()
for index, label in enumerate(ax2.get_xaxis().get_ticklabels()):
    xticks[index].set_visible(False)  # hide ticks where labels are hidden
print('done3')
#make a list of bins
no_bins = 100
bins = list(np.arange(sins.min().min(), sins.max().max(), int(abs(sins.min().min())+sins.max().max())/no_bins))
bins.append(sins.max().max())

# calculate histogram
hists = []
for col in sins.columns:
    count, division = np.histogram(sins.iloc[:,col], bins=bins)
    hists.append(count)
hists = pd.DataFrame(hists, columns=[str(i) for i in bins[1:]])
print('done4')

cmap = mpl.colors.ListedColormap(['white', '#FFFFBB', '#C3FDB8', '#B5EAAA', '#64E986', '#54C571',
          '#4AA02C', '#347C17', '#347235', '#25383C', '#254117'])

#heatmap
im = ax3.pcolor(hists.T, cmap=cmap)
cbar = plt.colorbar(im, cax=ax4)

yticks = np.arange(0, len(bins))
yticklabels = hists.columns.tolist()
ax3.set_yticks(yticks)
ax3.set_yticklabels([round(i,1) for i in bins], fontsize = 'xx-small')
ax3.set_title('Count')
yticks = ax3.yaxis.get_major_ticks()

for index, label in enumerate(ax3.get_yaxis().get_ticklabels()):
    if index % 3 != 0: #make some labels invisible
        yticks[index].set_visible(False)  # hide ticks where labels are hidden

plt.show()





