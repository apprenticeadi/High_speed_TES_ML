import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.utils import DataUtils
from src.traces import Traces
from src.ML_funcs import return_artifical_data
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

multiplier = 3
num_bins = 1000
guess_peak = 30
pca_components = 2  # it's really doubtful if pca helps at all
pca_cleanup = True
n_neighbors = 3
random_state = 0

frequency = 500

data_high, filtered_label = return_artifical_data(frequency,3)
X_train, X_test, y_train, y_test = train_test_split(data_high, filtered_label)

dim = len(data_high[0])
n_classes = len(np.unique(filtered_label))

# Reduce dimension to 2 with PCA
pca = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=random_state))

# Reduce dimension to 2 with LinearDiscriminantAnalysis
lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=3))

# Reduce dimension to 2 with NeighborhoodComponentAnalysis
nca = make_pipeline(
    StandardScaler(),
    NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),
)

# Use a nearest neighbor classifier to evaluate the methods
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
rf = RandomForestClassifier()
# Make a list of the methods to be compared
dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]
dim_reduction_methods = [("PCA", pca), ("LDA", lda)]

# plt.figure()
save3D = False
for i, (name, model) in enumerate(dim_reduction_methods):
    print(name)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # plt.subplot(1, 3, i + 1, aspect=1)

    # Fit the method's model
    model.fit(X_train, y_train)
    print('fitting done')
    # Fit a nearest neighbor classifier on the embedded training set
    knn.fit(model.transform(X_train), y_train)
    rf.fit(model.transform(X_train), y_train)
    # Compute the nearest neighbor accuracy on the embedded test set
    acc_knn = knn.score(model.transform(X_test), y_test)
    acc_rf = rf.score(model.transform(X_test), y_test)
    print(acc_knn, acc_rf)
    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = model.transform(data_high)
    print('transformation done')
    frames = []
    if save3D:
        def update_frame(angle):
            ax.view_init(30, angle)
            ax.cla()  # Clear the previous frame
            ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=filtered_label, s=30, cmap="Set1")
            ax.set_title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn))
        print('making animation')
        ani = animation.FuncAnimation(fig, update_frame, frames=range(0, 360), interval=50)
        ani.save('PCA_plot.gif', writer='pillow')
        print('animation saved')
    else:
        ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=filtered_label, s=30, cmap="Set1")
        ax.set_title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn))

        plt.show()