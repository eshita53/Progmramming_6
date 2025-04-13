import dask.dataframe as dd
import dask.delayed
import yaml
from data_loading_dask import retrieve_data
import time
from utility_functions.helping_functions import args_parser, profile_memory_usage, append_to_csv, sub_classification, create_file_with_directory
from utility_functions.evaluator import Evaluator
from feature_processor_dask import FeatureProcessing
from dask_ml.model_selection import train_test_split
from dask.distributed import performance_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dask_helping_functions import setup_dask_client
import xgboost.dask as xgbDask
import numpy as np
import logging


@dask.delayed
def get_config(file_name):
    with open(file_name, 'r', encoding="UTF-8") as stream:
        config = yaml.safe_load(stream)
    return config


def main():
    
    args = args_parser()
    config_file = args.file
    config = get_config(config_file).compute()
    lung_file_name = config['lung3']
    gene_file_name = config['gene']
    output_file = config['mem_time_profiling']
    save_roc_curve = config['dask_roc_file_path']
    log_file = config['log_file_dask']
    report_path = config['report']

    create_file_with_directory(log_file)
    logging.basicConfig(filename=log_file, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    # elapsed time calculation
    start_time = time.time()
    # memory allocation calculation during the execution of data loading
    mem_usage, dataset = profile_memory_usage(
        retrieve_data, lung_file_name, gene_file_name)
    elapsed_time_data_retrieval = time.time() - start_time

    logger.info("Dataset Loaded")

    # client to set up the local dask scheduler, enabling parallel execution and access to the dashboard
    client = setup_dask_client(n_workers=4,
                               threads_per_worker=2,
                               memory_limit='20GB')

    with performance_report(filename=report_path):
        y = dataset['characteristics.tag.histology'].map(
            sub_classification, meta=('output_column', 'str'))
        logger.info("Sub-classification Done")
        dataset = dataset.drop(columns='characteristics.tag.histology')
        encoder = LabelEncoder()
        y_encoded = dd.from_array(encoder.fit_transform(y),
                                columns=['classes']).classes
        logger.info("Target class Encoded")

        train_X, test_X, train_y, test_y = train_test_split(
            dataset, y_encoded, test_size=0.2, random_state=42, shuffle=False)
        logger.info("Dataset splitted into train-test split")

        # elapsed time feature processing
        start_time_process = time.time()
        fp = FeatureProcessing(logger=logger)
        fp.fit(train_X, train_y)
        X_train = fp.transform(train_X)
        logger.info("Feature processing done: Train")
        X_test = fp.transform(test_X)
        logger.info("Feature processing done: Test")
        end_time_processing = time.time() - start_time_process

        train_y = train_y.to_dask_array(lengths=True)
        test_y = test_y.to_dask_array(lengths=True)

        dtrain = xgbDask.DaskDMatrix(client, X_train, train_y)
        dtest = xgbDask.DaskDMatrix(client, X_test)

        logger.info("train-test lazy loaded")
        xgb_pars = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42
        }
        # elapsed time model training
        logger.info("Model training start")
        start_time_model_training = time.time()

        booster = xgbDask.train(
            client,
            xgb_pars,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train')]
        )
        logger.info("Model Training Done")
        logger.info("Model Predict Start")
        end_time_model_training = time.time() - start_time_model_training
        predict_proba = xgbDask.predict(client, booster, dtest)
        pred = predict_proba.map_blocks(lambda x: np.argmax(x, axis=1), dtype=int)

        predict_proba = predict_proba.compute()
        pred = pred.compute()
        test_y = test_y.compute()

        logger.info("Model Predictions Done")

        # elapsed time model evaluation
        start_time_model_evaluation = time.time()
        f1_score, precision, recall = Evaluator.f1_score_recall_precision_calculation(
            test_y, pred)
        accuracy = Evaluator.eval(test_y, pred, print_result=False)
        auc_score = Evaluator.sklearn_roc_curve(test_y, predict_proba)
        Evaluator.plot_roc_curve(test_y, predict_proba, save_roc_curve)
        end_time_evaluation = time.time() - start_time_model_evaluation

        df = pd.DataFrame({
            "workflow_name": ["Dask"],
            "memory_usage_during_execution": [mem_usage],
            "elapsed_time_data_retrieval": [elapsed_time_data_retrieval],
            "elapsed_time_processing": [end_time_processing],
            "elapsed_time_model_training": [end_time_model_training],
            "elapsed_time_model_evaluation": [end_time_evaluation],
            "accuracy": accuracy,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
            "auc": auc_score

        })
        df_dask = dd.from_pandas(df, npartitions=1)

        append_to_csv(df_dask, output_file)
        logger.info("Report saved")


if __name__ == "__main__":
    main()
