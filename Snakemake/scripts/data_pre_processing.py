import pandas as pd
import pickle
from feature_processing import FeatureProcessing
from sklearn.preprocessing import LabelEncoder
from helping_function import setup_logging


def process_features(logger, input_train_X_file, input_train_y_file, input_test_X_file, input_test_y_file, output_model_file,
                     output_encoded_target_file, output_train_file, output_test_X_file, output_encoded_train_y, output_encoded_test_y):
    """
    This function processes training and testing datasets by loading data from CSV files, encoding 
    the target variable using label encoding, and applying feature transformations. It initializes 
    a feature processor, fits it to the training data, and transforms both training and testing datasets. 
    The function then saves the fitted feature processor, label encoder, and transformed datasets for future use.
    """
    
    try:
        train_X = pd.read_csv(input_train_X_file)
        train_y = pd.read_csv(input_train_y_file)
        test_X = pd.read_csv(input_test_X_file)
        test_y = pd.read_csv(input_test_y_file)
        logger.info(f"All training and test Data loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise FileNotFoundError(f"Failed to load data: {str(e)}")

    # Target varibale needs to be label encoded
    # as it is now catagorical
    LE = LabelEncoder()
    LE.fit(train_y)
    encoded_train_y = LE.transform(train_y)
    encoded_test_y = LE.transform(test_y)

    # Initialize and fit feature processor
    feature_processor = FeatureProcessing(quantile_percentage=95,
                                          nan_threshold=35,
                                          alpha=0.06)

    # Fit on training data
    feature_processor.fit(train_X, encoded_train_y)

    transformed_train_X = feature_processor.transform(train_X)
    transformed_test_X = feature_processor.transform(test_X)

    # Save the fitted feature processor for later use
    with open(output_model_file, 'wb') as f:
        pickle.dump(feature_processor, f)
    logger.info(f"Feature processor saved to {output_model_file}")

    # save the fitted label encoder for later use
    with open(output_encoded_target_file, 'wb') as f:
        pickle.dump(LE, f)
    logger.info(f"Label encoder saved to {output_encoded_target_file}")

    transformed_train_X.to_csv(output_train_file, index=False)
    transformed_test_X.to_csv(output_test_X_file, index=False)
    # transfored the encoded numpy array to pandas dataframe to save it as csv
    pd.DataFrame(encoded_train_y).to_csv(output_encoded_train_y, index=False)
    pd.DataFrame(encoded_test_y).to_csv(output_encoded_test_y, index=False)

    logger.info("Feature processing completed successfully.")


def main():
    logger = setup_logging(snakemake.params.log_file, snakemake.params.rule_name)

    process_features(logger, snakemake.input.train_X, snakemake.input.train_y, snakemake.input.test_X, snakemake.input.test_y,
                     snakemake.output.fitted_feature_processors, snakemake.output.fitted_label_encoded_train_y,
                     snakemake.output.transformed_train_X, snakemake.output.transformed_test_X,
                     snakemake.output.label_encoded_train_y, snakemake.output.label_encoded_test_y)


if __name__ == "__main__":
    main()
