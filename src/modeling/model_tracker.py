import json
from addict import Dict
import os

temp_folder = "temp/"


class ModelTracker():

    @staticmethod
    def create_tracker(algorithm, majority_cost):
        tracker = Dict()
        tracker.algorithm = algorithm
        tracker.majority_cost = str(majority_cost)
        tracker.iterations = []
        return Dict(tracker)

    @staticmethod
    def dump_tracker(obj):
        # make dir
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)

        _file = os.path.join(temp_folder, obj.algorithm + '_' + obj.majority_cost + '.json')
        # make remove file if exists
        if os.path.exists(_file):
            os.remove(_file)
        # write to file if exists
        with open(_file, 'w') as outfile:
            json.dump(obj.iterations, outfile)
