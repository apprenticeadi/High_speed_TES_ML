import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.ML_funcs import return_artifical_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

'''
script to find the first three pca components and plot them to show variance, turn save3D to True to save a £D gif file
'''

random_state = 0
save3D = False
frequency = 1000
power = 6

data_high, filtered_label = return_artifical_data(frequency,1.8,power)
X_train, X_test, y_train, y_test = train_test_split(data_high, filtered_label)

dim = len(data_high[0])
n_classes = len(np.unique(filtered_label))

'''
try both PCA and LDA (LinearDiscriminantAnalysis, https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf page 106) 
'''

pca = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=random_state))

lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=3))


dim_reduction_methods = [("PCA", pca), ("LDA", lda)]

for i, (name, model) in enumerate(dim_reduction_methods):

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    model.fit(X_train, y_train)


    X_embedded = model.transform(data_high)

    frames = []
    if save3D:
        def update_frame(angle):
            ax.view_init(30, angle)
            ax.cla()  # Clear the previous frame
            ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=filtered_label, s=30, cmap="Set1")
            ax.set_title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name))
        ani = animation.FuncAnimation(fig, update_frame, frames=range(0, 360), interval=50)
        ani.save('PCA_plot.gif', writer='pillow')

    else:

        ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=filtered_label, s=30, cmap="Set1")
        ax.set_title(f"first three {name} components, for frequency :{frequency}kHz, power:{power}")

        plt.show()