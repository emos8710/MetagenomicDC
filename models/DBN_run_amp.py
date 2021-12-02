import numpy as np
import sys
np.random.seed(1337)  # for reproducibility

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.classification import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification

data_file = ""
data = ""
model_file = "./results/model.pkl"
model = SupervisedDBNClassification.load(model_file)

pred = model.predict(data)
