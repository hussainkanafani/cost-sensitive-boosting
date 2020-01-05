import json
from addict import Dict
import os

temp_folder = "temp/"


class ModelTracker():

    @staticmethod
    def create_tracker(algorithm):
        tracker = Dict()
        tracker.algorithm = algorithm
        tracker.iterations = []
        return Dict(tracker)

    @staticmethod
    def dump_tracker(obj):
        # make dir
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)

        _file = temp_folder + obj.algorithm + '.json'
        # make remove file if exists
        if os.path.exists(_file):
            os.remove(_file)
        # write to file if exists
        with open(_file, 'w') as outfile:
            json.dump(obj.iterations, outfile)
