from helping_function import setup_logging
import pandas as pd


def main():

    logger = setup_logging(snakemake.params.log_file,
                           snakemake.params.rule_name)
    model_files = snakemake.input.metrics
    all_models_data = []
    logger.info("Extracting Each model evalution metrics")
    for file_path in model_files:
        model_df = pd.read_csv(file_path)
        all_models_data.append(model_df)

    combined_metrics = pd.concat(all_models_data, ignore_index=True)
    logger.info("Combined all metrics evaluation metrics")

    output_file = snakemake.output.combine_results
    combined_metrics.to_csv(output_file, index=False)

    logger.info(f"Combined metrics has been saved to {output_file}")


if __name__ == "__main__":
    main()
