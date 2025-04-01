from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas import DataFrame


class DataProcessor():

    def __init__(self, dataframe: DataFrame) -> None:
        self.dataframe: DataFrame = dataframe
        self.dataframe.reset_index(drop=True, inplace=True)

    def find_col_position(self, pattern):

        for i, col in enumerate(self.dataframe.columns):
            if col.startswith(pattern):
                return i

    def find_cols_on_type(self, dtype):

        return self.dataframe.select_dtypes(include=[dtype]).columns.tolist()

    def drop_columns(self, column_list):
        self.dataframe = self.dataframe.drop(columns=column_list)

    def drop_nan_columns(self, nan_threshold=35):
        cols = [k for k, v in self.percentage_missing_values().items()
                if v >= nan_threshold]
        self.removed_cols = cols
        self.dataframe = self.dataframe.drop(columns=cols)

    @staticmethod
    def find_cols_on_types(dataframe, dtype):
        return dataframe.select_dtypes(include=[dtype]).columns.tolist()

    @staticmethod
    def change_datatypes(dataframe, columns_list, dtype):
        # change the column datatype for a column list based on the dtype
        dataframe[columns_list] = dataframe[columns_list].apply(
            lambda x: x.astype(dtype))
        return dataframe

    @staticmethod
    def get_cols(dataframe):
        pattern = '"'
        start_pos = None
        for i, col in enumerate(dataframe.columns):
            if col.startswith(pattern):
                start_pos = i
                break
        end_pos = dataframe.columns.get_loc(dataframe.columns[-1])
        # change the column datatype from start po to end pos
        columns_to_convert = dataframe.columns[start_pos:end_pos+1]
        return columns_to_convert

    def change_column_datatype(self):

        start_pos = self.find_col_position('"')
        end_pos = self.dataframe.columns.get_loc(self.dataframe.columns[-1])
        # change the column datatype from start po to end pos
        columns_to_convert = self.dataframe.columns[start_pos:end_pos+1]
        self.dataframe[columns_to_convert] = self.dataframe[columns_to_convert].apply(
            pd.to_numeric, errors='coerce')

    def impute_not_available_values(self, column_name):
        self.dataframe[column_name] = self.dataframe[column_name].replace(
            "Not Available", np.nan)

    def rename_columns(self, column_position, new_name):
        # Rename a column by position
        column_name = self.dataframe.columns[column_position]
        self.dataframe = self.dataframe.rename(columns={column_name: new_name})

    def percentage_missing_values(self):
        return (self.dataframe.isna().mean() * 100).to_dict()

    def remove_non_related_columns(self):
        """Remove unrelated columns where all the values are same and they don't have any missing values. Also a 
        columns where all the values are unique and also don't have any 
        """
        for col in self.dataframe.columns:
            is_null_sum = self.dataframe[col].isnull().sum()
            len_unique_val = len(self.dataframe[col].unique())
            if is_null_sum == 0 and len_unique_val == 1:
                self.dataframe = self.dataframe.drop(columns=col)

    def cramerV(self, train_y, covariance_threshold=0):
        """
        Computes Cramér's V correlation for categorical variables against a target column.

        :param train_y: The target column as a Pandas Series.
        :param covariance_threshold: Minimum correlation value to filter results.
        """
        # Select only categorical columns
        categorical_cols = self.find_cols_on_type("object")
        df = self.dataframe[categorical_cols]

        # Convert train_y to DataFrame (if not already)
        train_y_df = train_y.to_frame() if isinstance(train_y, pd.Series) else train_y

        # Merge categorical features with the target variable
        categorical_df = df.merge(
            train_y_df, left_index=True, right_index=True)
        target_column = train_y.name

        # Initialize DataFrame to store correlations
        cramer = pd.DataFrame(index=categorical_df.columns,
                              columns=[target_column])

        # Compute Cramér's V correlation for each categorical column
        for column in categorical_cols:
            if column == target_column:
                cramer.loc[column, target_column] = 1.0
            else:
                v = self.helper_cramerV(
                    categorical_df[target_column], categorical_df[column])
                cramer.loc[column, target_column] = v

        # Fill NaN values with 0 and ensure float data type
        cramer.fillna(0, inplace=True)
        cramer = cramer.astype(float)

        # Filter correlations above threshold
        cramer_filtered = cramer[cramer[target_column] > covariance_threshold]

        # Store results in class attributes
        self.cramer_matrix = cramer_filtered.T
        self.covarrianced_columns = cramer_filtered[target_column].index.tolist(
        )

    def plot_cramer(self, target_column):

        plt.figure(figsize=(10, 2))
        sns.heatmap(self.cramer_matrix, annot=True, fmt='.2f',
                    cmap='coolwarm', linewidths=.5)
        plt.title(
            f"Cramers V Correlation of '{target_column}' with Other Categorical Columns")
        plt.xlabel("Columns")
        plt.ylabel("Cramers v Correlations")
        plt.xticks(rotation=45)
        plt.show()

    def selecting_high_variance_gene_expression(self, quantile_percentage=95):
        """
        percentage would be in integer like 95 like this.
        """

        gene_columns = [
            col for col in self.dataframe.columns if col.startswith('"')]
        df = self.dataframe[gene_columns]
        percentage = quantile_percentage / 100

        others_df = self.dataframe[list(
            set(self.dataframe.columns)-set(gene_columns))]
        gene_variance = df.var(axis=0)  # calculating variance for each column
        percentile_threshold = gene_variance.quantile(percentage)
        self.selected_genes = gene_variance[gene_variance >
                                            percentile_threshold].index
        filtered_df = df[self.selected_genes]

        self.dataframe = pd.concat([others_df.reset_index(
            drop=True), filtered_df.reset_index(drop=True)], axis=1)

    def encoding_catagorical_features(self, one_hot_encoder: OneHotEncoder, catagorical_columns):

        encoded = one_hot_encoder.transform(
            self.dataframe[catagorical_columns])
        features_name = one_hot_encoder.get_feature_names_out()
        filtered_df = pd.DataFrame(encoded, columns=features_name)
        self.dataframe = self.dataframe.drop(columns=catagorical_columns)
        self.dataframe = pd.concat([filtered_df.reset_index(
            drop=True), self.dataframe.reset_index(drop=True)], axis=1)

    def fit_standard_scaling(self, scaler: StandardScaler):
        """ starting position of numerical features and ending position of numerical value"""

        scaler.transform(self.dataframe[self.find_cols_on_type('float64')])

    @staticmethod
    def helper_cramerV(label, x):
        """Cramér's V correlation with bias correction."""
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
                    "Unable to calculate Cramér's V using bias correction. Consider not using bias correction",
                    RuntimeWarning
                )
                return 0
            else:
                return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

        except Exception as e:
            print("Error occurred:", e)
            return 0
