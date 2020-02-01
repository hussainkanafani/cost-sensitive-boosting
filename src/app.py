import json
import os
import numpy as np
from modeling.logger import Logger
from processing.data_processor import DataProcessor
from modeling.adacost import AdaCost
from modeling.utils import createClassifier, classes_ordered_by_instances
from modeling.model_runner import ModelRunner
from addict import Dict
from modeling.evaluator import evaluate
from modeling.visualiser import *


def main(config):
    logger = Logger.createLogger(config['logger'])
    logger.info('Starting Application')
    results = {}
    base_estimator = config.model.base_estimator
    n_estimators = config.model.n_estimators
    learning_rate = config.model.learning_rate
    random_state = config.model.random_state
    datasets = config.dataProcessor.datasets
    tracker = config.tracker
    n_experiments= config.n_experiments
    
    step = .1
    start = .1
    end = 1.
    cost_setup = np.arange(start, end, step)
    all_measures = {}

    for dataset in datasets:

        # iterate over algorithms
        for algorithm in config.model.algorithms:             
            all_measures[algorithm] = {}
            all_measures[algorithm]["fmeasures"] = []
            all_measures[algorithm]["precisions"] = []
            all_measures[algorithm]["recalls"] = []

            for experiment in range(n_experiments):
                logger.info('Processing dataset {} ...'.format(dataset.filename))

                # get splitted data
                dataProcessor = DataProcessor(dataset, config, logger)
                classes = classes_ordered_by_instances(dataProcessor.data['trainY'])
                                            
                fmeasures = []
                precisions = []
                recalls = []

                for _cost in cost_setup:
                    class_weight = {
                    # minority class
                    classes[0]: 1,
                    # majority class
                    classes[1]: _cost
                    }

                    # create model
                    model = createClassifier(
                        algorithm, base_estimator, n_estimators, learning_rate, class_weight, np.random.randint(1000), tracker)
                    model_runner = ModelRunner(model, dataProcessor.data)

                    logger.info('Training algorithm {} ...'.format(algorithm))
                    # fit model
                    results[algorithm] = model_runner.run()
                    
                    # evaluate
                    fmeasure, _, precision, recall = evaluate(
                                    dataProcessor.data['testY'].tolist(),
                                    results[algorithm].tolist()
                                    )

                    fmeasures.append(fmeasure)
                    precisions.append(precision)
                    recalls.append(recall)

                # measures for all experiments
                all_measures[algorithm]["fmeasures"].append(fmeasures)
                all_measures[algorithm]["precisions"].append(precisions)
                all_measures[algorithm]["recalls"].append(recalls)

            all_measures[algorithm]["avg_fmeasure"] = np.mean(all_measures[algorithm]["fmeasures"],axis=0)                
            all_measures[algorithm]["avg_precision"] = np.mean(all_measures[algorithm]["precisions"],axis=0)
            all_measures[algorithm]["avg_recall"] = np.mean(all_measures[algorithm]["recalls"],axis=0)
            
            logger.info("all_measures {}".format(all_measures))
            plot_cost_fmeasure_gmean(algorithm,cost_setup, all_measures[algorithm]["avg_fmeasure"], all_measures[algorithm]["avg_precision"],all_measures[algorithm]["avg_recall"]).show()


def readAppConfigs():
    with open('./config-dev.json') as f:
        config = json.load(f)
    config = Dict(config)
    config.app.rootDir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))

    return config


def get_measures_by_name(measures, name):
    res = []
    for element in measures:
        res.append(element[name])
    return res


if __name__ == "__main__":
    main(readAppConfigs())
