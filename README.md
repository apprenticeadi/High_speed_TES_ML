# Machine learning for TES signal classification
### package requirements:
1. tensorflow
2. keras
3. sklearn
4. tsai
5. xgboost

### using the code:

6. if more data is taken try so save in the filename format ie 'raw_x/ykHz_yyyy-m-d_hh-mm-ss.txt', this is in the matlab script anyway.

7. Data files are processed similarly in the utils file, however with a slightly new method. two traces per row were taken so this was split for powers above 5
the method to read high frequency data was also edited to take the new format of data. to generate the artificial data a new method was put into the traces class
called generate high frequency data. this is then paired with the return artificial data method in the ML file which returns both the data and the labels. a function
to remove the offset is also included in the ML file which just returns the difference in peak height for the average artificial and actual trace.
8. Machine learning:
run the 'Overlap_ML' file, setting the relevant parameters at the top such as modeltype, and power. this will loop through all repetition rates
assuming you have data for 200-1000. it will then produce a bar plot for each rep rate and save the probabilities in a parameters file.

using the ML class takes fairly standard commands:
- initialise the model with the dataset and the labels, it is also possible to specify the modeltype here.
- train the model using the makemodel() function.
- the predict(X) function takes a set of traces X, usually the actual data and returns a numpy array of each corresponding predicted label
- there are also some commands that return some metrics about the model such as a confusion matrix and classification report which can be very useful
to perform a feature extraction on an individual trace call the extract_features function in the ML file. adding or removing features is straightforward, just add to the end
of the feature vector the function returns and the rest of the code does not need altering.

a good way of visualising the feature space and the feature importances is using the feature_exploration file.

9. Deep Learning
to use the deep learning or state of the art hybrid approaches, run the CNN or IT_model file where you set the parameters at the top.
the cnn structure is created by adding layers ie model.add(...) more details of this can be found in the tensorflow or keras documentation.
the state of the art models were all implemented using the tsai package so specific detail about models can be found on there. Its not an easy
package and has quite old requirements of other packages so may need to downgrade numpy.

10. Tomography routine
the probability values are stored in the parameters file after running either Overlap_ML, CNN or IT_test, so this is first loaded in. other parameters are loaded
from the log file. the tomography routine is then run. to change the probability values to that of another model, uncomment or change the filepath. 



