
from .adacost import AdaCost
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier



def createClassifier(algorithm, base_estimator, n_estimators, learning_rate, class_weight, random_state):
    if base_estimator == "SVC":
        base_estimator = svm.SVC(gamma=2, C=1)
    elif base_estimator == "KNN":
        base_estimator = KNeighborsClassifier(3)
    else:
        base_estimator = None

    return AdaCost(base_estimator, n_estimators, learning_rate, algorithm, class_weight, random_state)
