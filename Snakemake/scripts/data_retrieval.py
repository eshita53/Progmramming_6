import yaml
import pandas as pd
import numpy as np
from helping_function import setup_logging
from sklearn.model_selection import train_test_split

# doing sub classification for the targeted columns as they have 17 uniques class
# two big group and rest of them are mostly one. that's why sub dividing into two groups
def sub_classification(histology):
    if "Carcinoma" in histology:
        return 'Carcinoma'
    elif "Adenocarcinoma" in histology:
        return 'Adenocarcinoma'
    else:
        return 'Others'


def retrieve_data(lung3_file_name, gene_file_name):
    lung3_df = pd.read_csv(lung3_file_name)
    gene = pd.read_csv(gene_file_name, sep='\t', comment='!')
    gene.set_index('ID_REF', inplace=True)
    gene = gene.T
    gene.reset_index(drop=True, inplace=True)
    combined_df = lung3_df.merge(gene, left_index=True, right_index=True)
    return combined_df


def main():

    logger = setup_logging(snakemake.params.log_file,
                           snakemake.params.rule_name)
    try:
        combine_df = retrieve_data(
            snakemake.input.lung_file, snakemake.input.gene_file)

    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise FileNotFoundError(f"Failed to load data: {str(e)}")

    # Doing sub classifications
    combine_df['classes'] = combine_df['characteristics.tag.histology'].apply(
        lambda x: sub_classification(x))
    combine_df = combine_df.drop(columns='characteristics.tag.histology')

    # Before doing any preprocessing steps we will split the data into train and test in order to prevent data leakage
    train_X, test_X, train_y, test_y = train_test_split(
        combine_df.iloc[:, :-1],  # Features
        combine_df.iloc[:, -1],   # Target variable
        random_state=42
    )
    logger.info(f"Data has been split successfully")

    train_X.to_csv(snakemake.output.train_X, index=False)
    test_X.to_csv(snakemake.output.test_X, index=False)
    train_y.to_csv(snakemake.output.train_y, index=False)
    test_y.to_csv(snakemake.output.test_y, index=False)

    logger.info(f"Train and test data has been saved successfully")


if __name__ == "__main__":
    main()
