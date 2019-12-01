import json
from modeling.Logger import Logger

def main():

    with open('./config.json') as f:
        config = json.load(f)

    logger = Logger.createLogger(config['logger'])
    logger.info('Starting Application')

if __name__ == "__main__":
    main()