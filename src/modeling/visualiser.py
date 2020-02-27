import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from PIL import Image

def plot_cost_fmeasure_precision_recall(algorithm, costs, fmeasures, precisions, recalls):
    """

    plots fmeasures, precisions, and recalls regarding the costs
    
    algorithm: name of the algorithm as string
    costs: list of used costs to compute the metrics
    fmeasures: list of computed fmeasures. The length equals len(costs)
    precisions: list of computed precisions. The length equals len(costs)
    recalls: list of computed recalls. The length equals len(costs)

    """

    plt.clf()
    sns.set()
    plt.plot(costs, fmeasures, '--bo', label='F-measure')
    plt.plot(costs, precisions, '--go', label='Precision')
    plt.plot(costs, recalls, '--ro', label='Recall')
    plt.legend(loc="upper left")
    plt.ylim(0,1)
    plt.xlabel('Cost Setup')    
    plt.ylabel('Metric Value')
    plt.title(algorithm)
    return convert_plot_to_object(plt)

def plot_fmeasure_imbalance_ratios(algorithm, ratios_setup, ratios_cost_setup, fmeasures):
    """

    plots the fmeasure of different costs regarding different imbalance ratios

    algorithm: name of the algorithm as string
    ratios_setup: list of ratios used to compute the fmeasures
    ratios_cost_setup: list of costs used to compute the fmeaures
    fmeasures: numpy array of shape (ratios_cost_setup, ratios_setup)

    """

    plt.clf()
    sns.set()

    for i in range(len(ratios_cost_setup)):
        plt.plot(ratios_setup, fmeasures[i], '--o', label='Cost {}'.format(ratios_cost_setup[i]))
        
    plt.legend(loc="upper left")
    plt.ylim(0,1)
    plt.xlabel('Imbalance Ratio')    
    plt.ylabel('F-measure')
    plt.title(algorithm)
    return convert_plot_to_object(plt)

def plot_stacked_barchart_weights_iterations(minority_weight_sums, majority_weight_sums, algorithm, cost):
    """

    plots a stacked barchart to show the behavior of an algorithm regarding its weight sum over the iterations

    minority_weight_sums: list of weights sums of the minority class, the length equals number of iterations
    majority_weight_sums: list of weights sums of the majority class, the length equals number of iterations
    algorithm: name of the algorithm as string
    cost: the used cost as string

    """
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
    """ converts matplotlib object into a PIL image object """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)

    return im