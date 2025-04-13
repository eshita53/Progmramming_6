from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
from scipy.stats import chi2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessors():

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def find_col_position(self, pattern):
        for i, col in enumerate(self.dataframe.columns):
            if col.startswith(pattern):
                return i

    def impute_notavailable_values(self, column_name):
        self.dataframe[column_name] = self.dataframe[column_name].replace("Not Available", np.nan)
    @staticmethod
    def percantage_missing_values(dataframe):
        return (dataframe.isna().mean() * 100).to_dict()
    @staticmethod
    def imputing_column_with_average(dataframe, column_name):
        if column_name in dataframe.columns:
            mean_value = round(dataframe[column_name].mean(), 2)  # Compute mean & round to 2 decimals
            dataframe[column_name] = dataframe[column_name].fillna(mean_value)
        else:
            print(f"Column '{column_name}' not found in DataFrame.")
        return dataframe
    @staticmethod
    def transform_standard_scaling(dataframe, scaler: StandardScaler):
        
        """ starting position of numerical features and ending postionof numerical value"""
        cols = DataProcessors.find_cols_on_types(dataframe, 'number')
        dataframe[cols] = scaler.transform(dataframe[cols])
        return dataframe
    
    def selecting_high_variance_gene_expression(self, quantile_percentage=95):
        """
        percentage would be in integer like 95 like this.
        """

        gene_columns =  DataProcessors.find_cols_on_types(self.dataframe, 'number')
        
        df = self.dataframe[gene_columns]
        
        percentage = quantile_percentage / 100
        
        others_df = self.dataframe[list(set(self.dataframe.columns)-set(gene_columns))]
        gene_variance = df.var(axis=0)# calcualting variance for each column
        percentile_threshold = gene_variance.quantile(percentage)
        self.selected_genes = gene_variance[gene_variance>percentile_threshold].index
        filtered_df = df[self.selected_genes]
        self.dataframe = pd.concat([others_df.reset_index(drop=True), filtered_df.reset_index(drop=True)], axis=1)
        
    
    def dimention_reductions(self, model_type = 'PCA', n_component=2):
        if model_type == 'MDS':
            embedding = MDS(n_components = n_component)
            self.dimention_reduced_df = embedding.fit_transform(self.dataframe.select_dtypes(include = 'number'))
        elif model_type == 'PCA':
             embedding = PCA(n_components = n_component)
             self.dimention_reduced_df = embedding.fit_transform(self)
    
    @ staticmethod
    def encoding_catagorical_features (dataframe, one_hot_encoder: OneHotEncoder, catagorical_columns):

        encoded = one_hot_encoder.transform(dataframe[catagorical_columns])
        features_name = one_hot_encoder.get_feature_names_out(catagorical_columns)
        filtered_df = pd.DataFrame(encoded.toarray(), columns=features_name)
        dataframe = dataframe.drop(columns=catagorical_columns)
        dataframe = pd.concat([filtered_df.reset_index(drop=True), dataframe.reset_index(drop=True)], axis=1)
        return dataframe
        
    def remove_nonrelated_columns(self):
        """Remove unrelated columns where all the values are same and they don't have any misisng values. Also a 
        columns where all the values are unique and also don't have any 
        """
        for col in self.dataframe.columns:
            is_null_sum = self.dataframe[col].isnull().sum()
            len_unique_val = len(self.dataframe[col].unique())
            if is_null_sum == 0 and len_unique_val == 1:
                self.dataframe = self.dataframe.drop(columns=col)
    @staticmethod  
    def target_column_label_encoding(self,label_encoder : LabelEncoder, y):
        return label_encoder.transform(y)
    
    
    def drop_nan_columns(self, nan_threshold = 35):
        cols = [k for k,v in self.percantage_missing_values(self.dataframe).items() if v>=nan_threshold]
        self.removed_cols = cols
        self.dataframe = self.dataframe.drop(columns=cols)
    
    @staticmethod
    def change_datatypes(dataframe, columns_list, dtype):
        # chage the column datatype for a column list based on the dtype
        dataframe[columns_list] = dataframe[columns_list].apply(
            lambda x: x.astype(dtype))
        return dataframe

    def categorical_feat_sel_using_chi2(self, train_y, alpha):
        """select categorical features and remove others catagorical features from dataframe
        """
        cat_cols = DataProcessors.find_cols_on_types(self.dataframe, 'object')
        cat_df = self.dataframe[cat_cols]
        chi2_results = DataProcessors.calculating_chi2(cat_df, train_y)
        significant_feat = DataProcessors.significant_features(chi2_results, alpha)
        self.significant_features_cat = significant_feat
        remove_cols = [cols for cols in cat_cols if cols not in significant_feat]
        self.dataframe.drop(columns=remove_cols, inplace=True)
        
    
    @staticmethod
    def find_cols_on_types(dataframe, dtype):
        return dataframe.select_dtypes(include=[dtype]).columns.tolist()
    


    @staticmethod
    def calculating_chi2(df, train_y):
        """
        You have to ensure all the features are catagorical
        """

        chi2_results = {}

        for feature in df.columns:
            # Creating the contingency table
            contingency_table = pd.crosstab(df[feature], train_y)

            # calculating expected frequency
            row_totals = contingency_table.sum(axis=1)
            col_totals = contingency_table.sum(axis=0)
            grand_total = contingency_table.values.sum()

            expected = np.outer(row_totals, col_totals) / grand_total

            # computing the Chi-Square statistics(x^2)
            observed = contingency_table.values
            chi2_stat = ((observed - expected) ** 2 / expected).sum()

            # calculating degree of freedom
            dof = (len(row_totals) - 1) * (len(col_totals) - 1)

            # calculating p-value
            p_value = chi2.sf(chi2_stat, dof)  # Survival function (1 - CDF)

            # Storeing result for each features
            chi2_results[feature] = {
                'chi2': chi2_stat, 'dof': dof, 'p-value': p_value}

        return chi2_results

    @staticmethod
    def significant_features(chi2_result, alpha):
        significant_features = []
        for key, value in chi2_result.items():

            if value['p-value'] < alpha:
                significant_features.append(key)

        return significant_features
    
    def cramerV(self, train_y, coraviance_threshold=0):
            
            """ creamers v for catagorical variables
            , train_y is the targeted series"""

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
                    v = DataProcessors.helper_cramerV(catagorical_df[target_column], catagorical_df[column])
                    cramer.loc[column, target_column] = v

            # Fill NaN values with 0
            cramer.fillna(value=0, inplace=True)

            # Ensuring all the values are in float
            cramer = cramer.astype(float)

            # filtered the one which has corelations greater than corvariance_threshold
            cramer_filtered = cramer[cramer[target_column] > coraviance_threshold]

            self.cramer_matrix = cramer_filtered.T 
            
            self.covarrianced_columns = cramer_filtered[target_column].index.to_list()
            
            # return self
        
    def plot_cramer(self,target_column):
            plt.figure(figsize=(10, 2))
            sns.heatmap(self.cramer_matrix, annot=True, fmt='.2f',
                        cmap='coolwarm', linewidths=.5)
            plt.title(f"Cramers V Correlation of '{target_column}' with Other Catagorical Columns")
            plt.xlabel("Columns")
            plt.ylabel(" Cramers v Correlations")
            plt.xticks(rotation=45)
            plt.show()
