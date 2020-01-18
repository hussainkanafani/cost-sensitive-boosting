import json
import os
import numpy as np
from modeling.Logger import Logger
from processing.DataProcessor import DataProcessor
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
    
    all_measures=dict()
    for dataset in datasets:
        logger.info('Processing dataset {} ...'.format(dataset.filename))
        dataProcessor = DataProcessor(dataset, config, logger)
        # get splitted data
        data = [dataProcessor.data['trainX'], dataProcessor.data['trainY'],
                dataProcessor.data['testX'], dataProcessor.data['testY']]
        classes = classes_ordered_by_instances(dataProcessor.data['trainY'])
        step = .1
        start = .1
        end = 1.
        cost_setup = np.arange(start, end, step)
        # iterate over algorithms
        for algorithm in config.model.algorithms:             
            all_measures[algorithm]=dict()
            all_measures[algorithm]["fmeasures"]=[]
            all_measures[algorithm]["gmeans"]=[]
            for experiment in range(n_experiments):
                fmeasures=[]
                gmeans=[]               
                for _cost in cost_setup:
                    class_weight = {
                    # minority class
                    classes[0]: 1,
                    # majority class
                    classes[1]: _cost
                    }
                    # create model
                    model = createClassifier(
                        algorithm, base_estimator, n_estimators, learning_rate, class_weight, random_state, tracker)
                    model_runner = ModelRunner(model, data)
                    logger.info('Training algorithm {} ...'.format(algorithm))
                    # fit model
                    results[algorithm] = model_runner.run()
                    # predicted values
                    logger.info(
                        'Predicted Values of X-test {}'.format(results[algorithm].tolist()))
                    # predicted values
                    logger.info(
                        'True Values of X-test {}'.format(dataProcessor.data['testY'].tolist()))
                    # evaluate model                    
                    fmeasure,gmean = evaluate(dataProcessor.data['testY'].tolist(), results[algorithm].tolist())                    
                    fmeasures.append(fmeasure)
                    gmeans.append(gmean)
                # measures for all experiments
                all_measures[algorithm]["fmeasures"].append(fmeasures)
                all_measures[algorithm]["gmeans"].append(gmeans)
            all_measures[algorithm]["avg_gmean"] = np.mean(all_measures[algorithm]["gmeans"],axis=0)
            all_measures[algorithm]["avg_fmeasure"] = np.mean(all_measures[algorithm]["fmeasures"],axis=0)                
            logger.info("all_measures {}".format(all_measures))
            plot_cost_fmeasure_gmean(algorithm,cost_setup, all_measures[algorithm]["avg_fmeasure"], all_measures[algorithm]["avg_gmean"]).show()


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
