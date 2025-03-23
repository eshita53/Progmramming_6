from sklearn.ensemble import RandomForestClassifier
from helping_function import setup_logging
import pandas as pd
import pickle
def main():
    logger = setup_logging(snakemake.params.log_file,
                           snakemake.params.rule_name)
    
    train_X = pd.read_csv(snakemake.input.transformed_train_X)
    train_y = pd.read_csv(snakemake.input.label_encoded_train_y)
    
    random_forest_model = RandomForestClassifier(
    random_state=42, 
    n_estimators= 100,
    max_depth =10,
    min_samples_split =2,
    class_weight='balanced' # as the class is imbalanced
    )
    random_forest_model.fit(train_X,train_y)
    logger.info("Logistic Regression model training completed")
    
    output_path = snakemake.output[0]
    
    logger.info(f"Saving trained model to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(random_forest_model, f)
    
    logger.info("Saved the model Successfully")

if __name__ == "__main__":
    main()