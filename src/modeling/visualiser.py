import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.markers
import numpy as np
import io
from PIL import Image

def plot_cost_fmeasure_precision_recall(algorithm,costs, fmeasures,precision,recall):
    plt.clf()
    sns.set()
    plt.plot(costs, fmeasures, '--bo', label='F-measure')
    plt.plot(costs, precision, '--go', label='Precision')
    plt.plot(costs, recall, '--ro', label='Recall')
    plt.legend(loc="upper left")
    plt.ylim(0,1)
    plt.xlabel('Cost Setup')    
    plt.ylabel('Metric Value')
    plt.title(algorithm)
    return convert_plot_to_object(plt)

def plot_fmeasure_imbalance_ratios(algorithm, ratios_setup, ratios_cost_setup, fmeasures):
    plt.clf()
    sns.set()

    for i in range(len(ratios_cost_setup)):
        plt.plot(ratios_setup, fmeasures[i], '--o', label='Cost {}'.format(ratios_cost_setup[i]))

    plt.ylim(0,1)
    plt.xlabel('Imbalance Ratio')    
    plt.ylabel('F-measure')
    plt.title(algorithm)
    return convert_plot_to_object(plt)

def plot_instances_classes_weights_in_iteration(instances, classes, weights):
    plt.clf()
    COLOR_MAP = 'RdBu'
    sns.set()
    instancesAfterPca = PCA(n_components=2).fit_transform(instances)
    markers = ['o','*','x','D','p','s']
    cmap = matplotlib.cm.get_cmap(COLOR_MAP)
    #plot type according to classes, color according to weight
    for i in range(len(instancesAfterPca)):
        plt.plot(instancesAfterPca[i,0], instancesAfterPca[i,1], marker=markers[int(classes[i])], c=cmap(weights[i]))
    #show legend
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    #plt.show()
    return convert_plot_to_object(plt)

def plot_stacked_barchart_weights_iterations(minority_weight_sums, majority_weight_sums, algorithm, cost):
    plt.clf()
    ind = np.arange(len(minority_weight_sums))
    width = 0.35
    p1 = plt.bar(ind, minority_weight_sums, width=width, color='r')
    p2 = plt.bar(ind, majority_weight_sums, width=width,
             bottom=minority_weight_sums, color='b')
    plt.ylabel('Samples Weight Sum')
    plt.xlabel('Iteration')
    plt.title('{} (Majority Cost: {})'.format(algorithm, cost))
    plt.xticks(ind)
    plt.legend((p1[0], p2[0]), ('Minority', 'Majority'))

    return convert_plot_to_object(plt)

def convert_plot_to_object(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    #buf.close()

    return im
#plot_instances_classes_weights_in_iteration(np.random.randint(0,5,(10,5)),np.random.randint(0,6,10) , np.linspace(0,1,10))
#plot_stacked_barchart_weights_iterations(np.array([0.1, 0.0002, 0.3]), np.array([0.005, 0.01, 0.2]), 'adac1')