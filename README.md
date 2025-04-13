# Progmramming-6
This repository contains the portfolio assignment for the Programming-6 course. Each folder holds the solution for a specific assignment.
## Assignment Summaries
### [Assignment 1: Refactoring Machine Learning Workflow with Snakemake](https://github.com/eshita53/Progmramming_6/tree/main/Snakemake#refactoring-machine-learning-workflow-with-snakemake)
Snakemake is a Python language-based human-readable workflow management tool that is used for reproducible and scalable data analysis [1]. This project refactors a machine learning-based solution using Snakemake to automate the process. The pipeline handles data retrieval, preprocessing, model training, evaluation, benchmarking, and reporting.
### [Assignment 2: Lung Cancer Classification with Dask and XGBoost](https://github.com/eshita53/Progmramming_6/tree/main/dask#performance-of-dask-and-pandas-on-lung-cancer-classification-using-xgboost)
A scalable machine learning pipeline has been developed for a lung cancer research project aimed at classifying tumor subtypes using gene expression data and clinical features. Dask is utilized to process the large-scale dataset efficiently. We also used Dask with XGBoost to train a predictive model. The pipeline also includes an evaluation of the workflow’s computational efficiency to ensure scalability and performance.
### [Assignment 3: Assessing the Environmental Impact of ML Application](https://github.com/eshita53/Progmramming_6/tree/main/Green_ML#assessing-the-environmental-impact-of-ml-application)
Machine learning (ML) applications have rapidly grown in complexity and scale, raising concerns about their environmental impact. This project evaluates the carbon footprint of different ML approaches, focusing on data processing tools and model architecture choices.
## Installation 
To run this repository locally, follow these steps:
1. At first, clone this repository using this command: 
`git clone https://github.com/eshita53/Programming_6`
2. Then go to the folder that contains the repository contents.
`cd Programming_6`
3. The next step is to configure your environment. The Conda package manager is used in this tutorial. Please ensure that Conda or Miniconda are properly installed and configured on your system.
Run the following command inside the folder to create the environment:
`conda env create -f environment.yml`
4. Use the following command to activate the environment:
`conda activate programming_6`
5. Before executing the script ensure the environment is activated.

## Run
1. Open the folder for the assignment you want to execute.
2. Follow the instructions provided in that folder to run the specific assignment. Each folder contains details on how to execute the scripts related to that assignment.

## References
[1] Köster, J., & Rahmann, S. (2012). Snakemake--a scalable bioinformatics workflow engine. Bioinformatics (Oxford, England), 28(19), 2520–2522. doi:10.1093/bioinformatics/bts480




