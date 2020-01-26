import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn_pandas import CategoricalImputer

class DataProcessor:
    def __init__(self, dataset_config, config, logger):
        self.logger = logger
        self.dataset_config = dataset_config
        datasetPath = os.path.join(config['app']['rootDir'],
                                    config['dataProcessor']['dataDir'],
                                    self.dataset_config.filename)
        self.data = self.read_data(datasetPath)
        columns = self.data.columns.tolist()

        # replace question marks with -1 in order to fit KNNImputer
        self.data = self.replace_question_marks()
        self.data = self.impute_missing_values(columns)
        
        self.data = self.perform_one_hot_encoding()
        self.data = self.split_training_from_testing_set(config['dataProcessor']['testSetSize'])
        self.data = self.split_labels_from_data()

    def read_data(self, datasetPath):
        df = pd.read_csv(datasetPath)
        return df

    def perform_one_hot_encoding(self):
        for feature in self.dataset_config.categorical_features:
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
        # TODO: should it be a random construction ?
        self.logger.info('Splitting data ...')
        train, test = train_test_split(self.data, test_size=testSetSize)
        return {'train': train, 'test': test}

    def replace_question_marks(self):
        self.data = self.data.replace('?', -1)
        return self.data

    def impute_missing_values(self, columns):
        #imputer = KNNImputer(missing_values=-1, n_neighbors=1)
        imputer = CategoricalImputer(missing_values=-1, strategy='most_frequent')
        self.data = imputer.fit_transform(self.data.to_numpy())
    
        # converts numpy array to a pandas DataFrame
        self.data = pd.DataFrame(self.data, columns=columns)
        return self.data

    def ignore_missing_values(self):
        self.logger.info('Ignoring missing values ...')
        indicies = []
        for index, row in self.data.iterrows():
            if '?' in row.values:
                indicies.append(index)
        return self.data.drop(indicies)

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
        