from helping_function import setup_logging
import pandas as pd
from datetime import datetime
import base64
def encode_image(image_path):
    """Function to encode images to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def main():

    logger = setup_logging(snakemake.params.log_file,
                           snakemake.params.rule_name)
    
    combine_results = snakemake.input.combine_results
    benchmark_summary = snakemake.input.benchmark_summary
    roc_curve_plot = snakemake.input.roc_curve_plot
    accuracy_bar_chart = snakemake.input.accuracy_bar_chart
    f1_score_bar_chart = snakemake.input.f1_score_bar_chart
    total_run_time = snakemake.input.total_run_time
    report_output = snakemake.output.report
    
    logger.info("Extracting the data for generating reports ")
    
    result_df = pd.read_csv(combine_results)
    benchmark_summary_df = pd.read_csv(benchmark_summary)
    
    best_model = result_df.loc[result_df['accuracy'].idxmax()]['model_name']
    best_accuracy = result_df.loc[result_df['accuracy'].idxmax()]['accuracy']
    best_f1_score = result_df.loc[result_df['f1_score'].idxmax()]['f1_score']
    best_total_run_time = benchmark_summary_df[benchmark_summary_df['model_name'] == best_model]['total_runtime_sec'].values[0]
    
    # Encode all images
    roc_curve_encoded = encode_image(roc_curve_plot)
    accuracy_encoded = encode_image(accuracy_bar_chart)
    f1_score_encoded = encode_image(f1_score_bar_chart)
    total_run_time = encode_image(total_run_time)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Machine Learning Model Evaluation Report</title>
        <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .visualization {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .plot-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .plot {{ width: 100%; margin-bottom: 20px; }} in thi format
        </style>
    </head>
    <body>
        <h1>MACHINE LEARNING MODEL EVALUATION REPORT</h1>
        <p>Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>EXECUTIVE SUMMARY</h2>
            <p>This report summarizes the performance of {len(result_df)} machine learning models 
            trained for multiclass classification. The best performing model was <strong>{best_model}</strong> 
            with an accuracy of {best_accuracy:.2f}% and F1 score of {best_f1_score:.2f}. With the total runtime of {best_total_run_time:.2f} seconds.</p>
        </div>
        <h2>MODEL PERFORMANCE METRICS</h2>
        <table>
            <tr>
                {" ".join([f"<th>{col}</th>" for col in result_df.columns])}
            </tr>
            {"".join([f"<tr>{' '.join([f'<td>{value}</td>' for value in row])}</tr>" for row in result_df.values])}
        </table>
        <h2>BENCHMARK STATISTICS</h2>
        <table>
            <tr>
                {" ".join([f"<th>{col}</th>" for col in benchmark_summary_df.columns])}
            </tr>
            {"".join([f"<tr>{' '.join([f'<td>{value}</td>' for value in row])}</tr>" for row in benchmark_summary_df.values])}
        </table>
        
        <h2>VISUALIZATIONS</h2>
        <div class="plot-container">
            <div class="plot">
                <h3>ROC Curve</h3>
                <img src="data:image/png;base64,{roc_curve_encoded}" alt="ROC Curve">
            </div>
            
            <div class="plot">
                <h3>Accuracy Comparison</h3>
                <img src="data:image/png;base64,{accuracy_encoded}" alt="Accuracy Comparison">
            </div>
            
            <div class="plot">
                <h3>F1 Score Comparison</h3>
                <img src="data:image/png;base64,{f1_score_encoded}" alt="F1 Score Comparison">
            </div>
                        
            <div class="plot">
                <h3>Total Run Time Comparison</h3>
                <img src="data:image/png;base64,{total_run_time}" alt="Total Run Time Comparison">
            </div>
        </div>
        
        <h2>CONCLUSION </h2>
        <p>Based on the evaluation metrics, <strong>{best_model}</strong> is the recommended model for this classification task,
        demonstrating optimal performance with high accuracy and F1 score. The model achieves a good balance between
        predictive capability and computational efficiency.</p>
        
    </body>
    </html>
    """
    
    with open(report_output, 'w') as f:
        f.write(html)
    
    logger.info(f"Report generated successfully at {report_output}")

if __name__ == "__main__":
    main()
    
