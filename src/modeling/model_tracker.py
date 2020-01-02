import json
from addict import Dict
from pprint import pprint
from pymongo import MongoClient
import os

temp_folder="temp/"

class ModelTracker():
    
    @staticmethod
    def create_tracker(algorithm):
        tracker = Dict()
        tracker.algorithm = algorithm
        tracker.iterations = [] 
        return Dict(tracker)

    @staticmethod
    def dump_tracker(obj):
        _file= temp_folder + obj.algorithm + '.json'
        if os.path.exists(_file):
            os.remove(_file)
            print("delete "+_file+"\n")
                     
        with open(_file, 'w') as outfile:
            json.dump(obj.iterations, outfile)
