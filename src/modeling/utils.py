
from .adacost import AdaCost
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import numpy as np
import collections
import os


def createClassifier(algorithm, base_estimator, n_estimators, learning_rate, class_weight, random_state, root_path):
    if base_estimator == "DecisionTreeClassifier":
        base_estimator = tree.DecisionTreeClassifier(random_state=random_state, max_depth=5)
    else:
        raise ValueError('base estimator is not supported')

    return AdaCost(base_estimator, n_estimators, learning_rate, algorithm, class_weight, random_state, root_path)

def classes_ordered_by_instances(data):
    """ returns classes sorted by number of instances """
    classes = dict(collections.Counter(data.tolist()))
    keys,values=zip(*sorted(zip(classes.keys(),classes.values()),reverse=True))
    sorted_classes=keys
    return sorted_classes

def store_results(dir_path, file_name, pil_image):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_name += '.png'
    file_path = os.path.join(dir_path, file_name)
    pil_image.save(file_path)