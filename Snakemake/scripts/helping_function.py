import logging
def setup_logging(log_file, rule_name):
    
    # Configure the root logger only once
    if not logging.getLogger().handlers:
        # This will only execute the first time this function is called
        logging.basicConfig(
            level=logging.INFO,
            # for custom formatting of log used the format
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            # Used handelers to print both the console and log file
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    # Create a logger rule wise
    logger = logging.getLogger(rule_name)
    logger.setLevel(logging.INFO)
    
    return logger