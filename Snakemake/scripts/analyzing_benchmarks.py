from helping_function import setup_logging
import pandas as pd
import os
import re


def main():

    logger = setup_logging(snakemake.params.log_file,
                           snakemake.params.rule_name)
    file_list = snakemake.input
    output_file = snakemake.output.benchmark_summary

    logger.info("Extracting Each model training & evalution branchmark ")

    benchmark_data = []
    training = []
    evaluation = []

    for file in file_list:
        if re.match(r".*model_training_.*\.txt$", file):
            training.append(file)
        elif re.match(r".*model_evaluation_.*\.txt$", file):
            evaluation.append(file)

    # Process training benchmarks
    for bench_file in training:
        model_name = os.path.basename(bench_file).replace(
            "model_training_", "").replace(".txt", "")
        try:
            bench_df = pd.read_csv(bench_file, sep="\t")
            # Average runtime in seconds
            training_runtime = bench_df['s'].mean()
            benchmark_data.append({
                'model_name': model_name,
                'training_runtime_sec': round(training_runtime, 2),
                'evaluation_runtime_sec': None,
            })
        except Exception as e:
            print(
                f"Error processing training benchmark file {bench_file}: {str(e)}")
    for bench_file in evaluation:
        model_name = os.path.basename(bench_file).replace(
            "model_evaluation_", "").replace(".txt", "")
        try:
            bench_df = pd.read_csv(bench_file, sep="\t")
            eval_runtime = bench_df['s'].mean()  # Average runtime in seconds
            # Find the corresponding model in benchmark_data
            for model_data in benchmark_data:
                if model_data['model_name'] == model_name:
                    model_data['evaluation_runtime_sec'] = round(
                        eval_runtime, 2)
                    break
        except Exception as e:
            print(
                f"Error processing evaluation benchmark file {bench_file}: {str(e)}")

    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df['total_runtime_sec'] = benchmark_df['training_runtime_sec'] + \
        benchmark_df['evaluation_runtime_sec'].fillna(0)

    # Sort by total runtime (ascending)
    benchmark_df = benchmark_df.sort_values('total_runtime_sec')

    benchmark_df.to_csv(output_file, index=False)
    print(f"Benchmark summary saved to {output_file}")


if __name__ == "__main__":
    main()
