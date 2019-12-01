import json
import os
from modeling.Logger import Logger
from processing.DataProcessor import DataProcessor

def main():

    with open('./config.json') as f:
        config = json.load(f)
    config['app'] = {'rootDir': os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}

    logger = Logger.createLogger(config['logger'])
    logger.info('Starting Application')

    for dataset in config['dataProcessor']['datasets']:
        logger.info('Processing dataset {} ...'.format(dataset))
        dataProcessor = DataProcessor(dataset, config, logger)

if __name__ == "__main__":
    main()