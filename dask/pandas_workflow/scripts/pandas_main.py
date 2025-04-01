
import yaml
from data_loading_pandas import retrieve_data
import time
from utility_functions.helping_functions import args_parser, profile_memory_usage, sub_classification, create_file_with_directory
from utility_functions.evaluator import Evaluator
from pandas_workflow.scripts.feature_processing_pandas import FeatureProcessing
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import logging



def get_config(file_name):
    with open(file_name, 'r', encoding="UTF-8") as stream:
        config = yaml.safe_load(stream)
    return config


def main():
    
    args = args_parser()
    config_file = args.file
    config = get_config(config_file)
    lung_file_name = config['lung3']
    gene_file_name = config['gene']
    output_file  = config['mem_time_profiling']
    save_roc_curve = config['pandas_roc_file_path']
    log_file =  config['log_file_pandas']
    
    create_file_with_directory(log_file)
    logging.basicConfig(filename=log_file, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # elapsed time calculation
    start_time = time.time()
    mem_usage, dataset = profile_memory_usage(retrieve_data, lung_file_name, gene_file_name) # memory allocation calculation during the execution of data loading
    elapsed_time_data_retrieval = time.time()- start_time

    logger.info("Dataset Loaded")
    
    y = dataset['characteristics.tag.histology'].apply(lambda x: sub_classification(x))
    logger.info("Sub-classification Done")
    dataset = dataset.drop(columns='characteristics.tag.histology')
    
    encoder = LabelEncoder()
    encoder.fit(y)
    y_encoded = encoder.transform(y)
    y_encoded_series = pd.Series(y_encoded, name="encoded_labels")

    logger.info("Target class Encoded")
    train_X, test_X, train_y, test_y = train_test_split(dataset, y_encoded_series, test_size=0.2, random_state=42, shuffle=False)
    logger.info("Dataset splitted into train-test split")
    
    # elapsed time feature processing
    start_time_process = time.time()
    fp = FeatureProcessing()
    fp.fit(train_X, train_y)
    X_train = fp.transform(train_X)
    X_test = fp.transform(test_X)
    end_time_processing = time.time() - start_time_process
    logger.info("feature processing done")

    # elapsed time model training
    logger.info("Model training start")
    start_time_model_training = time.time()
    model = xgb.XGBClassifier(tree_method="hist")
    model.fit(X_train,train_y)
    end_time_model_training = time.time() - start_time_model_training
    logger.info("Model training Done")
    logger.info("Model Predict Start")
    pred = model.predict(X_test)
    predict_proba = model.predict_proba(X_test)
    logger.info("Model Prediction Done")
    
    logger.info("Model Evaluation Start")
    # elapsed time model evaluation
    start_time_model_evaluation = time.time()
    f1_score, precision, recall = Evaluator.f1_score_recall_precision_calculation(test_y, pred)
    accuracy = Evaluator.eval(test_y, pred, print_result = False)
    auc_score = Evaluator.sklearn_roc_curve(test_y,predict_proba)
    Evaluator.plot_roc_curve(test_y, predict_proba, save_roc_curve)
    end_time_evaluation = time.time() -  start_time_model_evaluation
    logger.info("Model Evaluation Done")

    df = pd.DataFrame({
        "workflow_name": ["Pandas"],
        "memory_usage_during_execution": [mem_usage],
        "elapsed_time_data_retrieval": [elapsed_time_data_retrieval],
        "elapsed_time_processing": [end_time_processing],
        "elapsed_time_model_training" :[end_time_model_training],
        "elapsed_time_model_evaluation": [end_time_evaluation],
        "accuracy": accuracy,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "auc": auc_score
        
    })
    
    df.to_csv(output_file, index=False)
    logger.info("Report saved")
    
if __name__ == "__main__":
    main()