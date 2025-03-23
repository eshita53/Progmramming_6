from helping_function import setup_logging
from sklearn.linear_model import LogisticRegression
import pandas as pd

import pickle
def main():
    logger = setup_logging(snakemake.params.log_file,
                           snakemake.params.rule_name)
    
    train_X = pd.read_csv(snakemake.input.transformed_train_X)
    train_y = pd.read_csv(snakemake.input.label_encoded_train_y)
    
    logistic_regression_model = LogisticRegression(
    random_state=42, 
    solver="saga",  # As we have large feature set & multiclass 
    multi_class="multinomial",  # As the target is multiclass
    penalty="l1",  # L1 helps select important features 
    C=0.1,  
    max_iter=1000, 
    n_jobs=-1  
    )
    logistic_regression_model.fit(train_X,train_y)
    logger.info("Logistic Regression model training completed")
    output_path = snakemake.output[0]
    logger.info(f"Saving trained model to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(logistic_regression_model, f)
    
    logger.info("Saved the model Successfully")

if __name__ == "__main__":
    main()