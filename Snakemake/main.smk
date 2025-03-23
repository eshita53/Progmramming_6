import os

def ensure_dirs(directories):
    """Ensure that all specified directories exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

configfile: "config.yaml"

# All the directories path
RAW_DATA_DIR = config['directories']['raw_data']
TRANSFORMED_DATA_DIR = config['directories']['transformed_data']
MODELS_DIR = config['directories']['models']
RESULTS_DIR = config['directories']['results']
PLOTS_DIR = config['directories']['plots']
LOGS_DIR = config['directories']['logs']
BENCHMARKS_DIR = config['directories']['benchmarks']
REPORTS_DIR = config['directories']['reports']
SCRIPTS_DIR = config['directories']['scripts']

# Required directories that need to be created
required_dir = [RAW_DATA_DIR,TRANSFORMED_DATA_DIR,MODELS_DIR,RESULTS_DIR,PLOTS_DIR,LOGS_DIR,BENCHMARKS_DIR,REPORTS_DIR]
# Ensure they exist before running Snakemake
ensure_dirs(required_dir)

# Common log file
LOG_FILE = f"{LOGS_DIR}/workflow.log"

rule all:
    input:
        expand(f"{RESULTS_DIR}/{{model}}_evaluation_metrics.csv", model=config['model_name']),
        f"{RESULTS_DIR}/benchmark_summary.csv",
        f"{RESULTS_DIR}/combine_results.csv",
        f"{PLOTS_DIR}/roc_curve.png",
        f"{PLOTS_DIR}/accuracy.png",
        f"{PLOTS_DIR}/f1_score.png",
        f"{REPORTS_DIR}/reports.html"

rule data_retrieval:
    input:
        lung_file=config['lung3'],
        gene_file=config['gene']
    output: 
        train_X=f"{RAW_DATA_DIR}/train_X.csv",
        test_X=f"{RAW_DATA_DIR}/test_X.csv",
        train_y=f"{RAW_DATA_DIR}/train_y.csv",
        test_y=f"{RAW_DATA_DIR}/test_y.csv"
    params:
        rule_name="data_retrieval",
        log_file=LOG_FILE
    script: 
        f"{SCRIPTS_DIR}/data_retrieval.py"

rule data_preprocessing:
    input:
        train_X=f"{RAW_DATA_DIR}/train_X.csv",
        train_y=f"{RAW_DATA_DIR}/train_y.csv",
        test_X=f"{RAW_DATA_DIR}/test_X.csv",
        test_y=f"{RAW_DATA_DIR}/test_y.csv"
    output:
        fitted_feature_processors=f"{MODELS_DIR}/feature_processor.pkl",
        fitted_label_encoded_train_y=f"{MODELS_DIR}/fitted_label_encoded_train_y.pkl",
        transformed_train_X=f"{TRANSFORMED_DATA_DIR}/transformed_train_X.csv",
        transformed_test_X=f"{TRANSFORMED_DATA_DIR}/transformed_test_X.csv",
        label_encoded_train_y=f"{TRANSFORMED_DATA_DIR}/label_encoded_train_y.csv",
        label_encoded_test_y=f"{TRANSFORMED_DATA_DIR}/label_encoded_test_y.csv"
    params:
        rule_name="data_preprocessing",
        log_file=LOG_FILE
    script:
        f"{SCRIPTS_DIR}/data_pre_processing.py"

rule model_training:
    input:
        transformed_train_X=f"{TRANSFORMED_DATA_DIR}/transformed_train_X.csv",
        label_encoded_train_y=f"{TRANSFORMED_DATA_DIR}/label_encoded_train_y.csv"
    output:
        f"{MODELS_DIR}/{{model}}.pkl"
    params:
        model_name=lambda wildcards: wildcards.model,
        rule_name=lambda wildcards: f"model_training_{wildcards.model}",  
        log_file=LOG_FILE
    benchmark: 
        f"{BENCHMARKS_DIR}/model_training_{{model}}.txt"
    script:
        f"{SCRIPTS_DIR}/{{wildcards.model}}.py"

rule model_evaluation:
    input:
        model=f"{MODELS_DIR}/{{model}}.pkl",
        test_X=f"{TRANSFORMED_DATA_DIR}/transformed_test_X.csv",
        test_y=f"{TRANSFORMED_DATA_DIR}/label_encoded_test_y.csv"
    output:
        metrics=f"{RESULTS_DIR}/{{model}}_evaluation_metrics.csv"
    params:
        model_name=lambda wildcards: wildcards.model,
        rule_name=lambda wildcards: f"model_evaluation_{wildcards.model}",
        log_file=LOG_FILE
    benchmark:
        f"{BENCHMARKS_DIR}/model_evaluation_{{model}}.txt"
    script:
        f"{SCRIPTS_DIR}/evaluate_model.py"

rule combine_results:
    input:
        metrics=expand(f"{RESULTS_DIR}/{{model}}_evaluation_metrics.csv", model=config['model_name'])
    output:
        combine_results=f"{RESULTS_DIR}/combine_results.csv"
    params:
        rule_name="combine_results",
        log_file=LOG_FILE
    script:
        f"{SCRIPTS_DIR}/combine_results.py"

rule analyzing_benchmarks:
    input:
        expand(f"{BENCHMARKS_DIR}/model_training_{{model}}.txt", model=config['model_name']) +
        expand(f"{BENCHMARKS_DIR}/model_evaluation_{{model}}.txt", model=config['model_name']) 
    output:
        benchmark_summary=f"{RESULTS_DIR}/benchmark_summary.csv"
    params:
        rule_name="analyzing_benchmarks",
        log_file=LOG_FILE
    script:
        f"{SCRIPTS_DIR}/analyzing_benchmarks.py"

rule visualize_performance:
    input:
        combine_results=f"{RESULTS_DIR}/combine_results.csv",
        benchmark_summary=f"{RESULTS_DIR}/benchmark_summary.csv"
    output:
        roc_curve_plot=f"{PLOTS_DIR}/roc_curve.png",
        accuracy_bar_chart=f"{PLOTS_DIR}/accuracy.png",
        f1_score_bar_chart=f"{PLOTS_DIR}/f1_score.png",
        total_run_time=f"{PLOTS_DIR}/run_time.png"
    params:
        rule_name="visualize_performance",
        log_file=LOG_FILE
    script:
        f"{SCRIPTS_DIR}/compare_model_performance.py"

rule generate_report:
    input: 
        combine_results=f"{RESULTS_DIR}/combine_results.csv",
        benchmark_summary=f"{RESULTS_DIR}/benchmark_summary.csv",
        roc_curve_plot=f"{PLOTS_DIR}/roc_curve.png",
        accuracy_bar_chart=f"{PLOTS_DIR}/accuracy.png",
        f1_score_bar_chart=f"{PLOTS_DIR}/f1_score.png",
        total_run_time=f"{PLOTS_DIR}/run_time.png"
    output:
        report=f"{REPORTS_DIR}/reports.html"
    params:
        rule_name="generate_report",
        log_file=LOG_FILE
    script:
        f"{SCRIPTS_DIR}/generate_report.py"