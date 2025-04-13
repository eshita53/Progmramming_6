# Performance of Dask and Pandas on Lung Cancer Classification using XGBoost

In this assignment, we evaluated whether using Dask provides any advantages over Pandas when handling a dataset with more than 66K columns. The objective was to perform lung cancer classification using XGBoost while leveraging Pandas and Dask for internal processing, such as feature engineering and data loading.  

Pandas is a powerful Python library widely used for data manipulation and analysis, but it operates in-memory, which can become a limitation for extremely large datasets. Dask, on the other hand, is designed for parallel computing and enables scalable data processing by breaking large datasets into smaller chunks that can be processed in parallel.

# Files and Directories  

## Configuration Files  
- **config.yaml** – Specifies raw dataset paths, machine learning models, and directories required for the project.  

## Python Files  

### Pandas Workflow  
- **scripts/data_loading_pandas.py** – Retrieves raw data and combines it.  
- **scripts/data_processor.py** – Contains the `DataProcessor` class for handling data processing tasks.  
- **scripts/feature_processing_pandas.py** – Defines a scikit-learn estimator for feature processing. It utilizes the `DataProcessor` class for data transformation.  
- **scripts/pandas_main.py** – Runs the full pandas workflow (Data Loading -> Feature transformation -> Model training -> Model Evaluation ).
- **pandas-workflow.ipynb** - Exploratory Data Analysis (EDA) workflow in pandas


### Dask Workflow  
- **scripts/data_loading_dask.py** – Retrieves raw data and combines it.  
- **scripts/data_processor_dask.py** – Contains the `DaskDataProcessor` class for handling data processing tasks.  
- **scripts/feature_processing_dask.py** – Defines a scikit-learn estimator for feature processing. It utilizes the `DataProcDaskDataProcessoressor` class for data transformation.  
- **scripts/dask_main.py** – Runs the full dask workflow (Data Loading -> Feature transformation -> Model training -> Model Evaluation ). 
- **dask-workflow.ipynb** - Exploratory Data Analysis (EDA) workflow in dask

### Utility Functions
- **evaluator.py** - Evaluator functions to measure the performance of the models
- **helping_functions.py** - some utility functions such as arg parser, memory profiling for both dask and pandas

### Other Directories
- **logs/** – Keeps track of workflow execution and errors.  
- **analysis/** – Stores final comparison plots on Pandas vs dask, along with a Python script to run this comparison.
- **reports/** – Stores final reports.

# Installation 
To run this repository locally, follow these steps:
1. At first, clone this repository using this command: 
`git clone https://github.com/eshita53/Programming_6`
2. Then go to the folder that contains the repository contents.
`cd Programming_6/dask`
3. The next step is to configure your environment. The Conda package manager is used in this tutorial. Please ensure that Conda or Miniconda are properly installed and configured on your system.
Run the following command inside the folder to create the environment:
`conda env create -f environment.yml`
4. Use the following command to activate the environment:
`conda activate dask_env`
5. Before executing the script, ensure the environment is activated.

# Run
If the `run.sh` script is not executable, make it executable by running this command:

```bash 
chmod +x run.sh
```

Then run the following to start the whole application

```bash 
./run.sh
```
