import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.markers
import numpy as np

def plot_cost_fmeasue_gmean(costs, fmeasues, gmean):
    sns.set()
    plt.plot(costs, fmeasues, '--bo', label='F-measure')
    plt.plot(costs, gmean, '--go', label='gmean')
    plt.legend(loc="upper left")
    plt.xlabel('Cost Setup')
    plt.ylabel('Metric Value')
    #plt.show()
    return plt

def plot_instances_classes_weights_in_iteration(instances, classes, weights):
    COLOR_MAP = 'RdBu'
    sns.set()
    instancesAfterPca = PCA(n_components=2).fit_transform(instances)
    markers = ['o','*','x','D','p','s']
    cmap = matplotlib.cm.get_cmap(COLOR_MAP)
    #plot type according to classes, color according to weight
    for i in range(len(instancesAfterPca)):
        plt.plot(instancesAfterPca[i,0], instancesAfterPca[i,1], marker=markers[classes[i]], c=cmap(weights[i]))
    #show legend
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    #plt.show()
    return plt

#plot_instances_classes_weights_in_iteration(np.random.randint(0,5,(10,5)),np.random.randint(0,6,10) , np.linspace(0,1,10))