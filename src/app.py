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

    for dataset in config['dataProcessor']['datasets']:
        logger.info('Processing dataset {} ...'.format(dataset['filename']))

        # iterate over algorithms (implicitly over experiments and costs)
        all_measures = loop_over_algorithms(dataset, logger, config)

        # store plots
        logger.info('Storing plots')
        for measure in all_measures.items():
            dataset_name = os.path.splitext(dataset['filename'])[0]
            dir_path = os.path.join(config['app']['rootDir'], 'data', 'results', dataset_name, config['model']['base_estimator'])

            if config['app']['imbalance_ratio']:
                # it has been iterated over ratios
                file_name = '{}_{}_measures_vs_ratios'.format(dataset_name, measure[0])
                store_results(dir_path, file_name, measure[1]['ratios_plot'])
            else:
                # it has been iterated over costs
                file_name = '{}_{}_measures_vs_costs'.format(dataset_name, measure[0])
                store_results(dir_path, file_name, measure[1]['measures_plot'])

                file_name = '{}_{}_weights_vs_iterations'.format(dataset_name, measure[0])
                store_results(dir_path, file_name, measure[1]['tracker_weights_plot'])

def loop_over_algorithms(dataset, logger, config):
    all_measures = {}

    for algorithm in config['model']['algorithms']:
        logger.info('\tTraining algorithm {}'.format(algorithm))
        
        # iterate over experiments (implicitly costs)     
        all_measures[algorithm] = loop_over_experiments(dataset, algorithm, logger, config)

        if config['app']['imbalance_ratio']:
            # it has been iterated over ratios
            all_measures[algorithm]["avg_ratios_fmeasures"] = np.swapaxes(np.mean(
                                                                            np.array(
                                                                            all_measures[algorithm]["ratios_fmeasures"]),
                                                                            axis=0),
                                                                            0, 1)
            ratios_plot = plot_fmeasure_imbalance_ratios(algorithm,
                                                        all_measures[algorithm]["ratios_accepted"],
                                                        config['app']['ratios_cost_setup'],
                                                        all_measures[algorithm]["avg_ratios_fmeasures"])
            all_measures[algorithm]["ratios_plot"] = ratios_plot
        else:
            # it has been iterated over costs
            all_measures[algorithm]["avg_fmeasure"] = np.mean(all_measures[algorithm]["fmeasures"], axis=0)                
            all_measures[algorithm]["avg_precision"] = np.mean(all_measures[algorithm]["precisions"], axis=0)
            all_measures[algorithm]["avg_recall"] = np.mean(all_measures[algorithm]["recalls"], axis=0)
            measures_plot = plot_cost_fmeasure_precision_recall(
                            algorithm,
                            config['app']['cost_setup'], 
                            all_measures[algorithm]["avg_fmeasure"], 
                            all_measures[algorithm]["avg_precision"],
                            all_measures[algorithm]["avg_recall"])
            all_measures[algorithm]["measures_plot"] = measures_plot

            # tracker weights plot for the cost 0.5
            # reads the weights of the trained algorithm in the last experiment from the dumped file in temp/
            cost = '_0.5'
            model_tracker_path = os.path.join(config['app']['rootDir'], 'src', 'temp', algorithm + cost +'.json')
            with open(model_tracker_path) as f:
                tracker_data = json.load(f)

            # prepare for computing sums
            sorted_classes = classes_ordered_by_instances(all_measures[algorithm]['trainY'])
            minority_weight_sums = []
            majority_weight_sums = []

            for iteration in tracker_data:
                # splitting weights
                # take weights of the instances that belongs to the minority or to the majority class
                sample_weight = np.array(iteration['sample_weight'])
                minority_weights = sample_weight[ all_measures[algorithm]['trainY'] == sorted_classes[0] ]
                majority_weights = sample_weight[ all_measures[algorithm]['trainY'] == sorted_classes[1] ]

                # computing avgs
                minority_weight_sums.append(np.sum(minority_weights))
                majority_weight_sums.append(np.sum(majority_weights))

            all_measures[algorithm]["tracker_weights_plot"] = plot_stacked_barchart_weights_iterations(minority_weight_sums,
                                                                                                majority_weight_sums,
                                                                                                algorithm,
                                                                                                cost[1:])

        logger.info("algorithm {} summary: {}".format(algorithm, all_measures))
    
    return all_measures

def loop_over_experiments(dataset, algorithm, logger, config):
    all_measures = {}
    all_measures["ratios_fmeasures"] = []
    all_measures["fmeasures"] = []
    all_measures["precisions"] = []
    all_measures["recalls"] = []
    
    # some ratios are not accepted by all datasets splits, because of the randomness and the very big/small size of the split
    all_ratios_accepted = []

    for experiment in range(config['app']['n_experiments']):
        logger.info('\t\texperiment {}'.format(experiment))
        # get splitted data
        dataProcessor = DataProcessor(dataset, config, logger)

        if config['app']['imbalance_ratio']:
            # iterate over ratios
            ratios_fmeasures, ratios_accepted = loop_over_ratios(algorithm, dataProcessor, logger, config)

            # fmeasures for all ratios
            all_measures["ratios_fmeasures"].append(ratios_fmeasures)
            all_ratios_accepted.append(ratios_accepted)
        else: 
            # iterate over costs
            fmeasures, precisions, recalls = loop_over_costs(algorithm, dataProcessor.data, logger, config)

            # measures for all experiments
            all_measures["fmeasures"].append(fmeasures)
            all_measures["precisions"].append(precisions)
            all_measures["recalls"].append(recalls)

    all_measures["trainX"] = dataProcessor.data['trainX']
    all_measures["trainY"] = dataProcessor.data['trainY']

    if config['app']['imbalance_ratio']:
        # drop experiments where there is less number of ratios experimented
        max_length = len(max(all_measures["ratios_fmeasures"], key=len))
        all_measures["ratios_fmeasures"] = [x for x in all_measures["ratios_fmeasures"] if len(x) == max_length]
        all_measures["ratios_accepted"] = max(all_ratios_accepted, key=len)

    return all_measures

def loop_over_costs(algorithm, data, logger, config):
    fmeasures = []
    precisions = []
    recalls = []
    classes = classes_ordered_by_instances(data['trainY'])

    for cost in config['app']['cost_setup']:
        class_weight = {
        # minority class
        classes[0]: 1,
        # majority class
        classes[1]: cost
        }

        # create model
        model = createClassifier(
            algorithm,
            config['model']['base_estimator'],
            config['model']['n_estimators'],
            config['model']['learning_rate'],
            class_weight,
            np.random.randint(1000),
            config['app']['rootDir']
            )

        model_runner = ModelRunner(model, data)

        logger.info('\t\t\t\tCost {}'.format(cost))
        predictions = model_runner.run()
        
        # evaluate
        fmeasure, precision, recall = evaluate(
                        data['testY'].tolist(),
                        predictions.tolist()
                        )

        fmeasures.append(fmeasure)
        precisions.append(precision)
        recalls.append(recall)

    return fmeasures, precisions, recalls

def loop_over_ratios(algorithm, dataProcessor, logger, config):
    ratios_fmeasures = []
    ratios_accepted = []

    for ratio in config['app']['ratio_setup']:
        try:
            data = dataProcessor.imbalance_data_using_rate(dataProcessor.data, ratio)
            ratios_accepted.append(ratio)
        except:
            # simply throw unaccepted ratios
            continue

        logger.info('\t\tRatio {}'.format(ratio))

        fmeasures = []
        classes = classes_ordered_by_instances(data['trainY'])
        for cost in config['app']['ratios_cost_setup']:
            
            class_weight = {
                # minority class
                classes[0]: 1,
                # majority class
                classes[1]: cost

            }
            # create model
            model = createClassifier(
                algorithm,
                config['model']['base_estimator'],
                config['model']['n_estimators'],
                config['model']['learning_rate'],
                class_weight,
                np.random.randint(1000),
                config['app']['rootDir']
                )

            model_runner = ModelRunner(model, data)

            logger.info('\t\t\tCost {}'.format(cost))
            predictions = model_runner.run()
            
            # evaluate
            fmeasure, _, _ = evaluate(
                            data['testY'].tolist(),
                            predictions.tolist()
                            )

            fmeasures.append(fmeasure)
        ratios_fmeasures.append(fmeasures)
    return ratios_fmeasures, sorted(list(set(ratios_accepted)))

def readAppConfigs():
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    with open(os.path.join(root_dir, 'src', 'config.json')) as f:
        config = json.load(f)

    config['app']['rootDir'] = root_dir

    return config

if __name__ == "__main__":
    main(readAppConfigs())
