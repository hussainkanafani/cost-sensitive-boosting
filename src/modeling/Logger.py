import logging 

class Logger:

    @staticmethod
    def createLogger(config):
        logger = logging.getLogger(config['name'])

        if config['verbose']:
            logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()

        if config['verbose']:
            ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter(config['format'])

        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger