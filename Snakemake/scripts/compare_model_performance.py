import pandas as pd
import matplotlib.pyplot as plt
import json
from helping_function import setup_logging
import numpy as np

def main():
    logger = setup_logging(snakemake.params.log_file,
                          snakemake.params.rule_name)
    
    model_data = pd.read_csv(snakemake.input.combine_results)
    benchmark_data = pd.read_csv(snakemake.input.benchmark_summary)
    logger.info('Extracting Model information')
 
    
    model_names = model_data['model_name'].unique()
    
    # ROC Plot
    plt.figure(figsize=(10, 8))
    for model_name in model_names:
        model_row = model_data[model_data['model_name'] == model_name].iloc[0]
        fpr_micro = json.loads(model_row['fpr_micro'])
        tpr_micro = json.loads(model_row['tpr_micro'])
        auc_micro = model_row['auc_micro']
        
        plt.plot(fpr_micro, tpr_micro, 
                label=f'{model_name} (AUC = {auc_micro:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Micro-average)')
    plt.legend(loc='lower right')
    plt.savefig(snakemake.output.roc_curve_plot)
    plt.close()
    logger.info('Saved ROC curve plot')
    
    # F1 score bar chart
    plt.figure(figsize=(10, 6))
    f1_scores = [model_data[model_data['model_name'] == model]['f1_score'].values[0] 
                for model in model_names]
    
    bars = plt.bar(model_names, f1_scores)
    plt.ylim([0, 1])
    plt.ylabel('F1 Score')
    plt.title('F1 Scores by Model')
    
    # Adding value labels on each bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.savefig(snakemake.output.f1_score_bar_chart)
    plt.close()
    logger.info('Saved F1 score bar chart')
    
    # Accuracy bar chart
    plt.figure(figsize=(10, 6))
    accuracies = [model_data[model_data['model_name'] == model]['accuracy'].values[0] 
                 for model in model_names]
    
    bars = plt.bar(model_names, accuracies)
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Model')
    
    # Adding value labels on each bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.savefig(snakemake.output.accuracy_bar_chart)
    plt.close()
    logger.info('Saved accuracy plot')
    
    
    # Total Runtime bar chart
    plt.figure(figsize=(10, 6))
    total_run_time = [benchmark_data[benchmark_data['model_name'] == model]['total_runtime_sec'].values[0] 
                 for model in model_names]
    
    bars = plt.bar(model_names, total_run_time)
    plt.ylim([0, 1])
    plt.ylabel('Total run Time (training + predict + evaluation)')
    plt.title('Total Run time by Model')
    
    # Adding value labels on each bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.savefig(snakemake.output.total_run_time)
    plt.close()
    logger.info('Saved Total run time plot')

if __name__ == "__main__":
    main()