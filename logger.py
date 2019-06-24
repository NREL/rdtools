import logging
import sys

logging.root.handlers = []

def init(level=None):
    logLevel = level or logging.INFO
    logging.root.handlers = []
    logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s| %(message)s',
                        level=logLevel,
                        stream=sys.stdout)
