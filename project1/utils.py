import logging
import os
from datetime import datetime

def initLogging(filename):
    logdir = "logs"

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    timestamp = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
    
    log_filename = f"{filename}_{timestamp}.log"
    log_filepath = os.path.join(logdir, log_filename)
    
    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("Logging initialized")