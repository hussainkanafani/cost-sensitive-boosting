import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .impute import CategoricalImputer
from imblearn.over_sampling import RandomOverSampler

class DataProcessor:
    def __init__(self, dataset_config, config, logger):
        self.logger = logger
        self.dataset_config = dataset_config
        datasetPath = os.path.join(config['app']['rootDir'],
                                    config['dataProcessor']['dataDir'],
                                    self.dataset_config['filename'])
        self.data = self.read_data(datasetPath)
        columns = self.data.columns.tolist()

        # replace question marks with -1 in order to fit Imputer
        self.data = self.replace_question_marks()
        self.data = self.impute_missing_values(columns)

        self.data = self.perform_one_hot_encoding()
        self.data = self.split_training_from_testing_set(config['dataProcessor']['testSetSize'])
        self.data = self.split_labels_from_data()

    def read_data(self, datasetPath):
        df = pd.read_csv(datasetPath)
        return df

    def perform_one_hot_encoding(self):
        for feature in self.dataset_config['categorical_features']:
            unique_categories = self.get_unique_categories(feature)

            # create new is_category column for each unique category
            for category in unique_categories:
                column = self.create_new_column_for_category(category, feature)
                self.data['is_' + str(feature) + '_' + str(category)] = column

            # drop the categorical column since it is not to be used anymore 
            del self.data[feature]
    
        return self.data

    def split_labels_from_data(self):
        trainX = self.data['train'].iloc[:, 1:].astype('float')
        trainY = self.data['train']['class'].astype('float')
        testX = self.data['test'].iloc[:, 1:].astype('float')
        testY = self.data['test']['class'].astype('float')

        return {'trainY': trainY, 'trainX': trainX,
                'testY': testY, 'testX': testX}

    def split_training_from_testing_set(self, testSetSize):
        train, test = train_test_split(self.data, test_size=testSetSize)
        return {'train': train, 'test': test}

    def imbalance_data_using_rate(self, data, rate):
        randomOverSampler = RandomOverSampler(sampling_strategy=rate)
        train_X_resampled, train_y_resampled = randomOverSampler.fit_resample(data['trainX'], data['trainY'])
        test_X_resampled, test_y_resampled = randomOverSampler.fit_resample(data['testX'], data['testY'])
        return {'trainY': train_y_resampled, 'trainX': train_X_resampled,
                'testY': test_y_resampled, 'testX': test_X_resampled}


    def replace_question_marks(self):
        self.data = self.data.replace('?', np.nan)
        return self.data

    def impute_missing_values(self, columns):
        imputer = CategoricalImputer(missing_values=np.nan, strategy='most_frequent')
        self.data = imputer.fit_transform(self.data)
    
        # converts numpy array to a pandas DataFrame
        self.data = pd.DataFrame(self.data, columns=columns)
        return self.data

    def get_unique_categories(self, column_name):
        # gets unique values of a given column in the dataset
        return self.data[column_name].unique().tolist()

    def create_new_column_for_category(self, category, feature):
        length = len(self.data['class'])
        zeros = [0] * length
        for index, value in enumerate(self.data[feature]):
            if value == category:
                zeros[index] = 1
        return zeros
        