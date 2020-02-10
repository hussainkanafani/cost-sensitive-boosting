import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.markers
import numpy as np
import io
from PIL import Image

def plot_cost_fmeasure_gmean(algorithm,costs, fmeasues,precision,recall):
    plt.clf()
    sns.set()
    plt.plot(costs, fmeasues, '--bo', label='F-measure')
    plt.plot(costs, precision, '--go', label='Precision')
    plt.plot(costs, recall, '--ro', label='Recall')
    #plt.plot(costs, gmean, '--go', label='gmean')
    plt.legend(loc="upper left")
    plt.ylim(0,1)
    plt.xlabel('Cost Setup')    
    plt.ylabel('Metric Value')
    plt.title(algorithm)
    #plt.show()
    return convert_plot_to_object(plt)

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

def convert_plot_to_object(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    #buf.close()

    return im
#plot_instances_classes_weights_in_iteration(np.random.randint(0,5,(10,5)),np.random.randint(0,6,10) , np.linspace(0,1,10))