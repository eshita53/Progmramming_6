
# Files and Directories  

## Configuration Files  
- **config.yaml** – Specifies raw dataset paths, machine learning models, and directories required for the project.  

## Python Files  

### Pandas Workflow  
- **scripts/data_loading_pandas.py** – Retrieves raw data and combines them.  
- **scripts/data_processor.py** – Contains the `DataProcessor` class for handling data processing tasks.  
- **scripts/feature_processing_pandas.py** – Defines a scikit-learn estimator for feature processing. It utilizes the `DataProcessor` class for data transformation.  
- **scripts/pandas_main.py** – Runs the full pandas workflow (Data Loading -> Feature transformation -> Model training -> Model Evaluation ).
- **pandas-workflow.ipynb** - EDA workflow in pandas


### Dask Workflow  
- **scripts/data_loading_dask.py** – Retrieves raw data and combines them.  
- **scripts/data_processor_dask.py** – Contains the `DaskDataProcessor` class for handling data processing tasks.  
- **scripts/feature_processing_dask.py** – Defines a scikit-learn estimator for feature processing. It utilizes the `DataProcDaskDataProcessoressor` class for data transformation.  
- **scripts/dask_main.py** – Runs the full dask workflow (Data Loading -> Feature transformation -> Model training -> Model Evaluation ). 
- **dask-workflow.ipynb** - EDA workflow in dask