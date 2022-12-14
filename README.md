# TES_python
Method outline
Author: Ruidi Zhu

Here, I present an outline of the method used.

## Calibration
1.	Take 100kHz data (500 data points). 
2.	Calculate the mean trace and overlap historgam
3.	Fitting the histogram, identify all the traces that correspond to zero-photon 
4.	Calculate the mean voltage of all zero-photon traces (this is the offset value). Then subtract all the traces from this value, such that the average voltage of zeros-photon traces is 0 (i.e. the average of zero-photon overlap is 0). This is important for later analysis since we want a dynamic zero.
5.	Classifying traces into different photon numbers. We select those traces that have overlap values lie in the range of (mean +/- multiplier*std)  for each photon number. Where mean and std are the means and standard deviations of overlap values for each photon number data. Multiplier can be chosen by ourself (setting multiplier to be a small value can make the analysis more accurate, but at a cost of wasting lots of data).
6.	calculate the mean traces for each photon number (they will be used for subtraction in later stage)

## Analysis of Higher Frequency Data with Tail Subtraction
1.	Split the data into individual traces
2.	Calculate mean trace, shift the data such that the mean trace starts with minimum value
3.	Calculate overlaps, plot histogram, fit the histogram (only the 0-photon fit is important here)
4.	Calculate the mean voltage of all zero-photon traces (this is the offset value). Then subtract all the traces from this value, such that the average voltage of zeros-photon traces is 0 (i.e. the average of zero-photon overlap is 0). 
5.	Using the shifted data, plot and fit the histogram (only the 1-photon data are important here)
6.	Identify those 1-photon data from the fit, calculate the mean trace for 1-photon data
7.	Compare the 1-photon mean traces for 100kHz data and the higher-frequency data, we need to normalize the 100kHz data based on the height and peaking position of the higher-frequency data, such that they peak at the same position and have the same height
8.	Perform the tail subtraction
9.	Use the subtracted traces to plot the overlap histogram, this is the final stegosaurus 
