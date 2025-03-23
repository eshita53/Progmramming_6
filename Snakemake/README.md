# Refactoring Machine Learning Workflow with Snakemake
Snakemake is a Python language-based human-readable workflow management tool that is used for
reproducible and scalable data analysis [1]. This project refactors a machine learning-based solution using Snakemake to automate the process. The pipeline handles data retrieval, preprocessing, model training, evaluation, benchmarking, and reporting. 

# Files and Directories  

## Configuration Files  
- **config.yaml** – Specifies raw dataset paths, machine learning models, and directories required for the project.  

## Python Files  

### Data Handling  
- **data_retrieval.py** – Retrieves raw data, splits it into training and testing sets, and saves them.  
- **data_processor.py** – Contains the `DataProcessor` class for handling data processing tasks.  
- **feature_processing.py** – Defines a scikit-learn estimator for feature processing. It utilizes the `DataProcessor` class for data transformation.  
- **data_pre_processing.py** – Applies feature processing estimators, transforms the data, and saves the fitted feature model.  

### Model Training  
- **RandomForest.py** – Trains a Random Forest model and saves it for later predictions.  
- **MulticlassLogisticsRegression.py** – Trains a Multiclass Logistic Regression model and saves it for future use.  

### Model Evaluation and Analysis  
- **evaluate_model.py** – Loads trained models, makes predictions, computes evaluation metrics, and saves the results.  
- **compare_model_performance.py** – Uses evaluation metrics to generate performance visualizations.  
- **combine_results.py** – Merges all model evaluation metrics into a single CSV file.  
- **analyzing_benchmarks.py** – Analyzes model training and evaluation benchmarks and merge them into a CSV file.  

### Reporting  
- **generate_report.py** – Generates an HTML report summarizing model performance, including visualizations.  

### Utility Functions  
- **helping_functions.py** – Contains utility functions used throughout the workflow.  

## Directories  
- **raw_data/** – Stores raw datasets before processing.  
- **transformed_data/** – Holds cleaned and processed data.  
- **models/** – Saves trained models.  
- **results/** – Stores model performance metrics.  
- **plots/** – Contains graphs comparing model performance.  
- **logs/** – Keeps track of workflow execution and errors.  
- **benchmarks/** – Records system performance data.  
- **reports/** – Stores final reports.  
- **scripts/** – Contains Python scripts for different tasks.
- 
# Installation Instructions
1. **Clone the Repository and Set Up Dependencies**: Follow [these instructions](https://github.com/eshita53/Progmramming_6/tree/main?tab=readme-ov-file#progmramming_6) to clone the repository and set up the dependencies.
3. **Set Up Configuration**:
   - Edit the **`config.yaml`** file to specify the dataset path and other parameters.





















# References
[1] Köster, J., & Rahmann, S. (2012). Snakemake--a scalable bioinformatics workflow engine. Bioinformatics (Oxford, England), 28(19), 2520–2522. doi:10.1093/bioinformatics/bts480
