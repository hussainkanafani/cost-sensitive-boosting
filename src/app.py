import json
import os
from modeling.Logger import Logger
from processing.DataProcessor import DataProcessor
from modeling.adacost import AdaCost
from modeling.utils import createClassifier
from modeling.model_runner import ModelRunner
from addict import Dict


def main(config):
    logger = Logger.createLogger(config['logger'])
    logger.info('Starting Application')
    results = {}
    base_estimator = config.model.base_estimator
    n_estimators = config.model.n_estimators
    learning_rate = config.model.learning_rate
    class_weight = config.model.class_weight
    random_state = config.model.random_state
    datasets = config.dataProcessor.datasets
    tracker =config.tracker
    for dataset in datasets:
        logger.info('Processing dataset {} ...'.format(dataset.filename))
        dataProcessor = DataProcessor(dataset, config, logger)
        # get splitted data
        data = [dataProcessor.data['trainX'], dataProcessor.data['trainY'],
                dataProcessor.data['testX'], dataProcessor.data['testY']]
        for algorithm in config.model.algorithms:
            model = createClassifier(
                algorithm, base_estimator, n_estimators, learning_rate, class_weight, random_state, tracker)
            model_runner = ModelRunner(model, data)
            logger.info('Training algorithm {} ...'.format(algorithm))
            results[algorithm] = model_runner.run()
            logger.info('Predicted Values of X-test {}'.format(results[algorithm]))


def readAppConfigs():
    with open('./config.json') as f:
        config = json.load(f)
    config = Dict(config)
    config.app.rootDir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))

    return config


if __name__ == "__main__":
    main(readAppConfigs())
