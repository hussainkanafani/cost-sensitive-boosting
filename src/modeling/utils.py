
from .adacost import AdaCost
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import numpy as np
import collections


def createClassifier(algorithm, base_estimator, n_estimators, learning_rate, class_weight, random_state,tracker):
    if base_estimator == "SVC":
        base_estimator = svm.SVC(gamma=2, C=1)
    elif base_estimator == "KNN":
        base_estimator = KNeighborsClassifier(3)
    else:
        base_estimator = tree.DecisionTreeClassifier()

    return AdaCost(base_estimator, n_estimators, learning_rate, algorithm, class_weight, random_state, tracker)

# returns classes sorted by number of instances
def classes_ordered_by_instances(data):
    classes = collections.Counter(data)
    return sorted(classes)