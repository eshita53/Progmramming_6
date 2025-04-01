
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import warnings

class DaskDataProcessor:
    def __init__(self, dataframe: dd.DataFrame) -> None:
        self.dataframe: dataframe = dataframe
        # Convert to dask dataframe if not already
        if isinstance(dataframe, pd.DataFrame):
            self.dataframe = dd.from_pandas(dataframe, npartitions=1)
    
    def find_col_position(self, pattern):
        for i, col in enumerate(self.dataframe.columns):
            if col.startswith(pattern):
                return i
    
    def find_cols_on_type(self, dtype):
        return self.dataframe.select_dtypes(include=[dtype]).columns.tolist()
    
    def drop_columns(self, column_list):
        self.dataframe = self.dataframe.drop(columns=column_list)
        return self
    
    def percentage_missing_values(self):
        # Compute missing values percentage directly
        return  self.dataframe.map_partitions(lambda x: x.isna().mean()).compute() * 100
    
    def drop_nan_columns(self, nan_threshold=35):
        
        # Compute missing values first
        missing_vals = self.percentage_missing_values()
        # Now missing_vals is a computed pandas Series, safe to iterate
        cols = [k for k, v in missing_vals.items() if v >= nan_threshold]
        self.removed_cols = cols
        self.dataframe = self.dataframe.drop(columns=cols)
        return self
    
    @staticmethod
    def change_datatypes(dataframe, columns_list, dtype):
        dataframe[columns_list] = dataframe[columns_list].astype(dtype)
        return dataframe
    
    def change_column_datatype(self):
        start_pos = self.find_col_position('"')
        if start_pos is not None:
            columns_to_convert = self.dataframe.columns[start_pos:]
            # Convert columns in parallel using map_partitions
            self.dataframe[columns_to_convert] = self.dataframe[columns_to_convert].map_partitions(
                lambda df: df.apply(pd.to_numeric, errors='coerce')
            )
        return self
    
    def impute_not_available_values(self, column_name):
        if column_name not in self.dataframe.columns:
            return self
        self.dataframe[column_name] = self.dataframe[column_name].map_partitions(
            lambda x: x.replace("Not Available", np.nan)
        )
        return self
    
    def remove_non_related_columns(self):
        """Remove columns with all same values or all unique values efficiently using Dask"""
        # Calculate null sums and unique values for all columns at once
        null_sums = self.dataframe.map_partitions(lambda x: x.isnull().sum()).compute()
        unique_vals = self.dataframe.map_partitions(lambda x: x.nunique()).compute()
        
        # Find columns to drop using vectorized operations
        cols_to_drop = [col for col, (nulls, uniques) in 
                        zip(self.dataframe.columns, zip(null_sums, unique_vals)) 
                        if nulls == 0 and uniques == 1]
        
        if cols_to_drop:
            self.dataframe = self.dataframe.drop(columns=cols_to_drop)
        return self
        
    def cramerV(self, train_y, covariance_threshold=0):
        # Get categorical columns
        cat_cols = self.find_cols_on_type("object")
        df = self.dataframe[cat_cols]
        
        # Convert train_y to pandas if it's a dask series
        if isinstance(train_y, dd.Series):
            train_y = train_y.compute()
        
        df[train_y.name] = train_y
        # Initialize results dictionary
        cramer_values = {}
        
        # Compute Cramer's V for each column
        for col in df.columns:
            col_data = df[col].compute()  # Compute column data
            if col == train_y.name:
                cramer_values[col] = 0.0
            else:
                cramer_values[col] = self.helper_cramerV(train_y, col_data)
        
        # Create and filter DataFrame
        cramer = pd.DataFrame(cramer_values, index=[train_y.name])
        cramer_filtered = cramer[cramer.iloc[0] > covariance_threshold]
        
        self.cramer_matrix = cramer_filtered.T
        self.covarianced_columns = [col for col in df.columns if cramer_values[col]> covariance_threshold]
        return self

    
    def selecting_high_variance_gene_expression(self, quantile_percentage=95):
        gene_columns = [col for col in self.dataframe.columns if col.startswith('"')]
        
        if not gene_columns:
            return self
            
        # Calculate variances for gene columns
        variances = self.dataframe[gene_columns].var().compute()
        percentage = quantile_percentage / 100
        percentile_threshold = variances.quantile(percentage)
        selected_genes = variances[variances > percentile_threshold].index
        
        # Keep non-gene columns
        other_columns = list(set(self.dataframe.columns) - set(gene_columns))
        
        # Create new dataframe with selected columns
        self.selected_genes = selected_genes
        self.dataframe = self.dataframe[other_columns + list(selected_genes)]
        return self
        
    @staticmethod
    def helper_cramerV(label, x):
        """Compute Cramer's V correlation with bias correction"""
        try:
            confusion_matrix = pd.crosstab(label, x)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            r, k = confusion_matrix.shape
            
            phi2 = chi2 / n
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            
            if min((kcorr - 1), (rcorr - 1)) == 0:
                warnings.warn(
                    "Unable to calculate Cramer's V using bias correction.",
                    RuntimeWarning
                )
                return 0
            else:
                return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
        except Exception as e:
            print("Error occurred:", e)
            return 0