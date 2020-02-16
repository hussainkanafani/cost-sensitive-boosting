import json
import os
import numpy as np
from modeling.logger import Logger
from processing.data_processor import DataProcessor
from modeling.adacost import AdaCost
from modeling.utils import createClassifier, classes_ordered_by_instances, store_results
from modeling.model_runner import ModelRunner
from modeling.evaluator import evaluate
from modeling.visualiser import *


def main(config):
    logger = Logger.createLogger(config['logger'])
    logger.info('Starting Application')
    cost_setup = np.arange(0.1, 1, 0.1)

    for dataset in config['dataProcessor']['datasets']:
        logger.info('Processing dataset {} ...'.format(dataset['filename']))

        # iterate over algorithms (implicitly over experiments and costs)
        all_measures = loop_over_algorithms(dataset, cost_setup, logger, config)

        if config['verbose']:
            for measure in all_measures.items():
                measure[1]['plot'].show()

        # store plots
        logger.info('Storing plots')
        for measure in all_measures.items():
            dataset_name = os.path.splitext(dataset['filename'])[0]
            dir_path = os.path.join(config['app']['rootDir'], 'data', 'results', dataset_name, config['model']['base_estimator'])
            
            file_name = '{}_{}'.format(dataset_name, measure[0])
            store_results(dir_path, file_name, measure[1]['measures_plot'])

            file_name = '{}_{}_weights'.format(dataset_name, measure[0])
            store_results(dir_path, file_name, measure[1]['tracker_avg_plot'])

            dir_path = os.path.join(dir_path, measure[0] + '_tracker')
            for i, tracker_plot in enumerate(measure[1]['tracker_plots']):
                file_name = '{}_{}_iter_{}'.format(dataset_name, measure[0], i)
                store_results(dir_path, file_name, tracker_plot)

def loop_over_algorithms(dataset, cost_setup, logger, config):
    all_measures = {}

    for algorithm in config['model']['algorithms']:
        logger.info('\tTraining algorithm {}'.format(algorithm))
        
        # iterate over experiments (implicitly costs)     
        all_measures[algorithm] = loop_over_experiments(dataset, algorithm, cost_setup, logger, config)

        all_measures[algorithm]["avg_fmeasure"] = np.mean(all_measures[algorithm]["fmeasures"], axis=0)                
        all_measures[algorithm]["avg_precision"] = np.mean(all_measures[algorithm]["precisions"], axis=0)
        all_measures[algorithm]["avg_recall"] = np.mean(all_measures[algorithm]["recalls"], axis=0)
        measures_plot = plot_cost_fmeasure_gmean(
                        algorithm,
                        cost_setup, 
                        all_measures[algorithm]["avg_fmeasure"], 
                        all_measures[algorithm]["avg_precision"],
                        all_measures[algorithm]["avg_recall"])

        # model tracker
        with open(os.path.join(config['app']['rootDir'], 'src', 'temp', algorithm + '.json')) as f:
            tracker_data = json.load(f)

        tracker_plots = []

        # prepare for computing avgs
        sorted_classes = classes_ordered_by_instances(all_measures[algorithm]['trainY'])
        minority_weight_avgs = []
        majority_weight_avgs = []

        for iteration in tracker_data:
            sample_weight = np.array(iteration['sample_weight'])
            tracker_plot = plot_instances_classes_weights_in_iteration(all_measures[algorithm]['trainX'],
                                                                    all_measures[algorithm]['trainY'].tolist(),
                                                                    sample_weight)
            tracker_plots.append(tracker_plot)

            # splitting weights
            minority_weights = sample_weight[ all_measures[algorithm]['trainY'] == sorted_classes[0] ]
            majority_weights = sample_weight[ all_measures[algorithm]['trainY'] == sorted_classes[1] ]

            # computing avgs
            minority_weight_avgs.append(np.mean(minority_weights))
            majority_weight_avgs.append(np.mean(majority_weights))

        all_measures[algorithm]["measures_plot"] = measures_plot
        all_measures[algorithm]["tracker_plots"] = tracker_plots
        all_measures[algorithm]["tracker_avg_plot"] = plot_stacked_barchart_weights_iterations(minority_weight_avgs,
                                                                                              majority_weight_avgs,
                                                                                              algorithm)

        logger.info("algorithm {} summary: {}".format(algorithm, all_measures))
    
    return all_measures

def loop_over_experiments(dataset, algorithm, cost_setup, logger, config):
    all_measures = {}
    all_measures["fmeasures"] = []
    all_measures["precisions"] = []
    all_measures["recalls"] = []

    for experiment in range(config['n_experiments']):
        logger.info('\t\texperiment {}'.format(experiment))
        # get splitted data
        dataProcessor = DataProcessor(dataset, config, logger)

        # iterate over costs
        fmeasures, precisions, recalls = loop_over_costs(algorithm, cost_setup, dataProcessor.data, logger, config)

        # measures for all experiments
        all_measures["fmeasures"].append(fmeasures)
        all_measures["precisions"].append(precisions)
        all_measures["recalls"].append(recalls)

    all_measures["trainX"] = dataProcessor.data['trainX']
    all_measures["trainY"] = dataProcessor.data['trainY']
        
    return all_measures

def loop_over_costs(algorithm, cost_setup, data, logger, config):
    fmeasures = []
    precisions = []
    recalls = []
    classes = classes_ordered_by_instances(data['trainY'])

    for _cost in cost_setup:
        class_weight = {
        # minority class
        classes[0]: 1,
        # majority class
        classes[1]: _cost
        }

        # create model
        model = createClassifier(
            algorithm,
            config['model']['base_estimator'],
            config['model']['n_estimators'],
            config['model']['learning_rate'],
            class_weight,
            np.random.randint(1000),
            config['tracker'])

        model_runner = ModelRunner(model, data)

        logger.info('\t\t\tCost {}'.format(_cost))
        predictions = model_runner.run()
        
        # evaluate
        fmeasure, _, precision, recall = evaluate(
                        data['testY'].tolist(),
                        predictions.tolist()
                        )

        fmeasures.append(fmeasure)
        precisions.append(precision)
        recalls.append(recall)

    return fmeasures, precisions, recalls

def readAppConfigs():
    with open('./config-dev.json') as f:
        config = json.load(f)
    config['app']['rootDir'] = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))

    return config


def get_measures_by_name(measures, name):
    res = []
    for element in measures:
        res.append(element[name])
    return res


if __name__ == "__main__":
    main(readAppConfigs())
