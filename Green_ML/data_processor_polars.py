import re
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import polars as pl
import numpy as np
from sklearn.preprocessing  import StandardScaler, OneHotEncoder
from sklearn.manifold import MDS 
from sklearn.preprocessing import LabelEncoder
from polars import DataFrame

class DataProcessor():

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe: DataFrame = dataframe
        
    def find_col_position(self, pattern):
        
        for i, col in  enumerate(self.dataframe.columns):
            if col.startswith(pattern):
                return i
            
    def find_cols_on_type(self, dtype):
        
        return self.dataframe.select(pl.col(dtype)).columns
    
    def drop_columns(self, column_list):
        self.dataframe = self.dataframe.drop(column_list)
       

    def drop_nan_columns(self, nan_threshold = 35):
        cols = [k for k,v in self.percantage_missing_values().items() if v[0]>=nan_threshold]
        self.removed_cols = cols
        self.dataframe = self.dataframe.drop(cols)
        # return self
    
    @staticmethod 
    def find_cols_on_types(dataframe, dtype):
        return dataframe.select(pl.col(dtype)).columns
    
    def impute_notavailable_values(self, column_name):
        self.dataframe = self.dataframe.with_columns(pl.col(column_name).replace("Not Available", np.nan))
        # return self
    
    def rename_columns(self, column_position, new_name):
        # Rename a column by position
        column_name = self.dataframe.columns[column_position]
        self.dataframe = self.dataframe.rename({column_name: new_name})
        # return self

    def percantage_missing_values(self):
        # return (self.dataframe.isna().mean() * 100).to_dict()
        return self.dataframe.select([(pl.col(col).is_null().sum()/ self.dataframe.height)*100 for col in self.dataframe.columns]).to_dict(as_series=True)

    def remove_nonrelated_columns(self):
        """Remove unrelated columns where all the values are same and they don't have any misisng values. Also a 
        columns where all the values are unique and also don't have any 
        """
        columns_to_remove = []
        is_null_sum_df = self.dataframe.select([pl.col(col).is_null().sum() for col in self.dataframe.columns])
        len_unique_val_df = self.dataframe.select([pl.col(col).n_unique() for col in self.dataframe.columns])
        for col in self.dataframe.columns:
            is_null_sum = is_null_sum_df[col][0]
            len_unique_val = len_unique_val_df[col][0]
            if is_null_sum == 0 and len_unique_val == 1:
                columns_to_remove.append(col)
               
        self.dataframe = self.dataframe.drop(col)
        # return self
    
    def cramerV(self, train_y, coraviance_threshold=0):
        
        """ start_pos and end_pos of catagorical_variables, train_y is the targeted series"""
        
        #selecting only the catagorical features
        cat_cols = self.find_cols_on_type(pl.String)
        df = self.dataframe[cat_cols]
        # merging this with the targed features to get the features which is valuable for target
 
        df = df.with_columns(train_y.alias(train_y.name))

        self.covarrianced_columns = []
        # Compute Cramer's V for each column
        for col in df.columns:
            col_data = df[col] 
            if col == train_y.name:
                continue
            elif self.helper_cramerV(train_y, col_data) > coraviance_threshold:
                self.covarrianced_columns.append(col)
        return self
    
    def plot_cramer(self,target_column):
        
        plt.figure(figsize=(10, 2))
        sns.heatmap(self.cramer_matrix, annot=True, fmt='.2f',
                    cmap='coolwarm', linewidths=.5)
        plt.title(f"Cramers V Correlation of '{target_column}' with Other Catagorical Columns")
        plt.xlabel("Columns")
        plt.ylabel(" Cramers v Correlations")
        plt.xticks(rotation=45)
        plt.show()
    
    def selecting_high_variance_gene_expression(self, quantile_percentage=95):
        """
        percentage would be in integer like 95 like this.
        """

        gene_columns = [col for col in self.dataframe.columns if col.startswith('"')]
        df = self.dataframe.select(gene_columns)
        percentage = quantile_percentage / 100
        
        others_df = self.dataframe.select(list(set(self.dataframe.columns)-set(gene_columns)))
        gene_variance = df.var() # calcualting variance for each column
        percentile_threshold = gene_variance.select(pl.all().quantile(percentage)).row(0)[0]
        self.selected_genes = [
            col for col in df.columns if gene_variance[col].item() > percentile_threshold
        ]
        filtered_df = df.select(self.selected_genes)

        self.dataframe = others_df.hstack(filtered_df)

        # return self
    
    def encoding_catagorical_features (self, one_hot_encoder: OneHotEncoder, catagorical_columns):

        encoded = one_hot_encoder.transform(self.dataframe[catagorical_columns])
        features_names = one_hot_encoder.get_feature_names_out()
        filtered_df = pl.DataFrame(encoded)
        filtered_df.columns = features_names
        self.dataframe = self.dataframe.drop(catagorical_columns)
        self.dataframe = self.dataframe.hstack(filtered_df)
        # return self
        
    def fit_standard_scaling(self, scaler: StandardScaler):
        
        """ starting position of numerical features and ending postionof numerical value"""

        scaler.transform(self.dataframe[self.find_cols_on_type(pl.Float64)])
        # return self
    
    @staticmethod
    def helper_cramerV(label, x):
        """Creamers v corelations with bias correction """
        import pandas as pd
        confusion_matrix = pd.crosstab(label, x)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape

        phi2 = chi2 / n
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)

        try:
            if min((kcorr - 1), (rcorr - 1)) == 0:
                warnings.warn(
                    "Unable to calculate Cramer's V using bias correction. Consider not using bias correction",
                    RuntimeWarning
                )
                return 0
            else:
                return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

        except Exception as e:
            print("Error occurred:", e)
            return 0
    
    def target_column_label_encoding(self,target_column):
        LE = LabelEncoder()
        self.dataframe['target_col_label'] = LE.fit_transform(self.dataframe[target_column])

            
    # As we have more features then instances MDS would be perfect for this
    def dimention_reductions(self, n_component=2):
        embedding = MDS(n_components = n_component)
        self.dimention_reduced_df = embedding.fit_transform(self.dataframe.select_dtypes(include = 'number'))

    
    def plot_dimentioned_reduced_data(self, train_y):
        unique_classes = train_y.unique()
        num_classes =  len(unique_classes)
        colors = sns.color_palette("husl", num_classes)
        label_to_color = {label: colors[i] for i, label in enumerate(unique_classes)}
        plt.figure(figsize=(10, 8))

        for label in unique_classes:
            class_points = self.dimention_reduced_df[train_y == label]
            
            plt.scatter(
                class_points[:, 0],
                class_points[:, 1],
                label=label,         
                color=label_to_color[label],
                alpha=0.7
            )
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.colorbar(label='Class Label')  
        plt.title('Multi Dimentional scaling of the datasets')
        plt.show()

        
          
    
        
        
        
        