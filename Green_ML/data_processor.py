import re
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing  import StandardScaler, OneHotEncoder
from sklearn.manifold import MDS 
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame


class DataProcessor():

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe: DataFrame = dataframe
        self.dataframe.reset_index(drop=True, inplace = True)
        
    def find_col_position(self, pattern):
        
        for i, col in  enumerate(self.dataframe.columns):
            if col.startswith(pattern):
                return i
            
    def find_cols_on_type(self, dtype):
        
        return self.dataframe.select_dtypes(include=[dtype]).columns.tolist()
    
    def drop_columns(self, column_list):
        self.dataframe = self.dataframe.drop(columns=column_list)
       

    def drop_nan_columns(self, nan_threshold = 35):
        cols = [k for k,v in self.percantage_missing_values().items() if v>=nan_threshold]
        self.removed_cols = cols
        self.dataframe = self.dataframe.drop(columns=cols)
        # return self
    
    @staticmethod 
    def find_cols_on_types(dataframe,dtype):
        return dataframe.select_dtypes(include=[dtype]).columns.tolist()
    @staticmethod
    def change_datatypes(dataframe, columns_list, dtype):
        # chage the column datatype for a column list based on the dtype
        dataframe[columns_list] = dataframe[columns_list].apply(lambda x: x.astype(dtype))
        return dataframe
    @ staticmethod
    def get_cols(dataframe):
        pattern = '"'
        start_pos = None
        for i, col in  enumerate(dataframe.columns):
            if col.startswith(pattern):
                start_pos = i
                break
        end_pos = dataframe.columns.get_loc(dataframe.columns[-1])
        # chage the column datatype from start po to end pos
        columns_to_convert = dataframe.columns[start_pos:end_pos+1]
        return columns_to_convert
    def change_column_datatype(self):
        
        start_pos = self.find_col_position('"')
        end_pos = self.dataframe.columns.get_loc(self.dataframe.columns[-1])
        # chage the column datatype from start po to end pos
        columns_to_convert = self.dataframe.columns[start_pos:end_pos+1]
        self.dataframe[columns_to_convert] = self.dataframe[columns_to_convert].apply(pd.to_numeric, errors='coerce')
                                                                      
        # return self
    
    def impute_notavailable_values(self, column_name):
        self.dataframe[column_name] = self.dataframe[column_name].replace("Not Available", np.nan)
        # return self
    
    def rename_columns(self, column_position, new_name):
        # Rename a column by position
        column_name = self.dataframe.columns[column_position]
        self.dataframe = self.dataframe.rename(columns={column_name: new_name})
        # return self

    def percantage_missing_values(self):
        return (self.dataframe.isna().mean() * 100).to_dict()

    def remove_nonrelated_columns(self):
        """Remove unrelated columns where all the values are same and they don't have any misisng values. Also a 
        columns where all the values are unique and also don't have any 
        """
        for col in self.dataframe.columns:
            is_null_sum = self.dataframe[col].isnull().sum()
            len_unique_val = len(self.dataframe[col].unique())
            if is_null_sum == 0 and len_unique_val == 1:
                self.dataframe = self.dataframe.drop(columns=col)
        # return self
    
    def cramerV(self, train_y, coraviance_threshold=0):
        
        """ start_pos and end_pos of catagorical_variables, train_y is the targeted series"""
        
        # combined_df = self.dataframe.merge(train_y, left_index=True,right_index=True)
        #selecting only the catagorical features
        df = self.dataframe[self.find_cols_on_type("object")]
        # merging this with the targed features to get the features which is valuable for target
 
        catagorical_df = df.merge(train_y,left_index=True,right_index=True)
        target_column = train_y.name
        cramer = pd.DataFrame(index=catagorical_df.columns, columns=[target_column])
        
        # Calculating Cramer's V corelatins for the specified target column against all other catagorical columns
        for column in df.columns:
            if column == target_column:
                cramer.loc[column, target_column] = 1.0
            else:
                v = DataProcessor.helper_cramerV(catagorical_df[target_column], catagorical_df[column])
                cramer.loc[column, target_column] = v

        # Fill NaN values with 0
        cramer.fillna(value=0, inplace=True)

        # Ensuring all the values are in float
        cramer = cramer.astype(float)

        # filtered the one which has corelations greater than corvariance_threshold
        cramer_filtered = cramer[cramer[target_column] > coraviance_threshold]

        self.cramer_matrix = cramer_filtered.T 
        
        self.covarrianced_columns = cramer_filtered[target_column].index.to_list()
        
    
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
        df = self.dataframe[gene_columns]
        percentage = quantile_percentage / 100
        
        others_df = self.dataframe[list(set(self.dataframe.columns)-set(gene_columns))]
        gene_variance = df.var(axis=0)# calcualting variance for each column
        percentile_threshold = gene_variance.quantile(percentage)
        self.selected_genes = gene_variance[gene_variance>percentile_threshold].index
        filtered_df = df[self.selected_genes]

        self.dataframe = pd.concat([others_df.reset_index(drop=True), filtered_df.reset_index(drop=True)], axis=1)

        # return self
    
    def encoding_catagorical_features (self, one_hot_encoder: OneHotEncoder, catagorical_columns):

        encoded = one_hot_encoder.transform(self.dataframe[catagorical_columns])
        features_name = one_hot_encoder.get_feature_names_out()
        filtered_df = pd.DataFrame(encoded, columns=features_name)
        self.dataframe = self.dataframe.drop(columns=catagorical_columns)
        self.dataframe = pd.concat([filtered_df.reset_index(drop=True), self.dataframe.reset_index(drop=True)], axis=1)
        # return self
        
    def fit_standard_scaling(self, scaler: StandardScaler):
        
        """ starting position of numerical features and ending postionof numerical value"""

        scaler.transform(self.dataframe[self.find_cols_on_type('float64')])
        # return self
    
    @staticmethod
    def helper_cramerV(label, x):
        """Creamers v corelations with bias correction """
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
                # warnings.warn(
                #     "Unable to calculate Cramer's V using bias correction. Consider not using bias correction",
                #     RuntimeWarning
                # )
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

        
          
    
        
        
        
        