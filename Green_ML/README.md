# Assessing the Environmental Impact of ML Application
Machine learning (ML) applications have rapidly grown in complexity and scale, raising concerns about
their environmental impact. This project evaluates the carbon footprint of different ML approaches,
focusing on data processing tools and model architecture choices.

# Files and Directories  

## Configuration Files  
- **config.yaml** – Specifies raw dataset paths for the project.  

## Python Files  
- **data_processor.py** – Contains the `DataProcessor` class for handling data processing tasks for pandas.  
- **data_processor_polars.py** – Contains the `DataProcessor` class for handling data processing tasks for polars.
- **pandas-workflow.ipynb** – This notebook contains the workflow for feature processing using pandas and the computation of carbon emissions. 
- **polars-workflow.ipynb** – This notebook outlines the workflow for feature processing using Polars, and the computation of carbon emissions. 
- **digit_recog.ipynb** – This notebook contains various experiments comparing different models based on their carbon emissions.

## Report:
-**Green_ML_Report.pdf** - This document presents an assessment report on the carbon emissions of machine learning applications.

# Installation Instructions
1. **Clone the Repository and Set Up Dependencies**: Follow [these instructions](https://github.com/eshita53/Progmramming_6/tree/main?tab=readme-ov-file#progmramming_6) to clone the repository and set up the dependencies.
2. **Set Up Configuration**:
   - Edit the **`config.yaml`** file to specify the dataset path.

# Running the application

Run `pandas-workflow.ipynb` and `polars-workflow.ipynb` one after the other to analyze data processing impact. The results will be saved in `emission.csv.` Then, run `digit_recog.ipynb` to log emissions for comparing large model impact.