import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, filename, config, logger):
        self.logger = logger
        datasetPath = os.path.join(config['app']['rootDir'],
                                    config['dataProcessor']['dataDir'],
                                    filename)
        self.data = self.readData(datasetPath)
        self.data = self.splitTrainingFromTestingSet(config['dataProcessor']['testSetSize'])
        self.data = self.splitLabelsFromData()

    def readData(self, datasetPath):
        df = pd.read_csv(datasetPath)
        return df

    def splitLabelsFromData(self):
        # TODO: adjust regarding column name or position
        trainX = self.data['train'].iloc[:, 1:]
        trainY = self.data['train']['class']
        testX = self.data['test'].iloc[:, 1:]
        testY = self.data['test']['class']

        return {'trainY': trainY, 'trainX': trainX,
                'testY': testY, 'testX': testX}

    def splitTrainingFromTestingSet(self, testSetSize):
        # TODO: should it be a random construction ?
        self.logger.info('  Splitting data ...')
        train, test = train_test_split(self.data, test_size=testSetSize)
        return {'train': train, 'test': test}