import sys
import numpy as np
import pandas as pd
from Blackbox41 import blackbox41
from Blackbox42 import Blackbox42

# -------------------------------------------------------
blackbox = sys.argv[-1]
if blackbox == "blackbox41":
    bb = blackbox41
    name = "blackbox41"
elif blackbox == "blackbox42":
    bb = blackbox42
    name = "blackbox42"
else:
    print('invalid blackbox')
    sys.exit()
# -------------------------------------------------------


class Data:

    def __init__(self):

        global bb
        global name
        self.bb = bb
        self.name = name
        self.classes = []  # Array to store final results

    # -------------------------------------------------------

    def read_write(self):

        # Read data from input file
        X_train = pd.DataFrame(self.bb.ask())

        # Call classifier class, and store results
        obj_c = Classifier(X_train)
        self.classes = obj_c.predict()

        # Write output file
        submission = pd.DataFrame(self.classes)
        submission = submission.astype(int)
        submission.to_csv("results_" + self.name + ".csv", index=False, header=False)

# -------------------------------------------------------


class Classifier:

    def __init__(self, X_train):

        self.k = 4  # Num of clusters
        self.iterations = 100  # Num of iter to get optimal results
        self.X_train = X_train  # Input Data
        self.y_pred = [0 for _ in range(self.X_train.shape[0])]  # Predicted classes

    # -------------------------------------------------------

    def predict(self):
        """
        1. Assign k random centroids which serve as clusters
        2. Iterate 3,4,5 multiple times to get get good clusters
        3. Calculate min distance for each point to every cluster (centroid)
        4. Assign a cluster/class to each pint based on its lowest distance to all centroid
        5. Update centroid based mean per feature per cluster
        :return: final predicted clsuters/classes/labels
        """

        # Choose k random points as centroids
        centroids = self.X_train.sample(n=4, random_state=42)

        for _ in range(self.iterations):

            # stores all train data correspondig to a particular cluster
            cluster = {k: [] for k in range(self.k)}

            # Calculate centroid for each row of input
            for i in range(self.X_train.shape[0]):

                # Calculate Euclidean Distance from centroid to each point
                distances = self.euclidean(self.X_train.iloc[i, :].values, centroids)
                self.y_pred[i] = int(np.argmin(distances))  # assign cluster based on lowest distance

                # append all train data rows corresponding to a particular label
                cluster[self.y_pred[i]].append(self.X_train.iloc[i, :].values)

            # Update centroid based on mean values of each feature
            for k in range(self.k):
                centroids.iloc[k, :] = pd.DataFrame(cluster[k]).mean()

        return self.y_pred

    # -------------------------------------------------------

    def euclidean(self, x, c):
        """
        Calculates Euclidean Distance b/w each point and all k-centroids
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
        :param x: input feature, shape (,2)
        :param c: centroid coord, shape (k,2)
        :return: euclidean distances, shape (0, k)
        """
        x, c = np.asarray(x), np.asarray(c)
        return np.linalg.norm(x - c, axis=1)

# -------------------------------------------------------


if __name__ == '__main__':
    obj = Data()
    obj.read_write()

