
import dask.dataframe as dd
from sklearn.base import BaseEstimator, TransformerMixin
from dask_ml.preprocessing import Categorizer, DummyEncoder, StandardScaler
from data_processor_dask import DaskDataProcessor


class FeatureProcessing(TransformerMixin, BaseEstimator):
    def __init__(self, covariance_threshold=0.20, quantile_percentage=95, nan_threshold=35, logger=None):
        self.covariance_threshold = covariance_threshold
        self.quantile_percentage = quantile_percentage
        self.nan_threshold = nan_threshold
        self.logger = logger
        
    def fit(self, X, y=None):
        
        if not isinstance(X, dd.DataFrame):
            X = dd.from_pandas(X, npartitions=4)  
        
        data_processor = DaskDataProcessor(X)
        data_processor.dataframe = data_processor.dataframe.persist()
        self.logger.info("Feature Processing starts")
        data_processor.remove_non_related_columns()
        self.logger.info("Feature Processing: removed non related columns")
        data_processor.impute_not_available_values('characteristics.tag.grade')
        self.logger.info("Feature Processing: imputed not available columns")
        data_processor.drop_nan_columns(self.nan_threshold)
        self.logger.info("Feature Processing: dropped nan columns")
        
        # Compute Cramer's V and filter columns
        data_processor.cramerV(y, self.covariance_threshold)
        self.logger.info("Feature Processing: compute cramer V")
        self.covarianced_columns = data_processor.covarianced_columns
        removed_categorical_features = set(data_processor.find_cols_on_type('object')) - set(self.covarianced_columns)
        data_processor.drop_columns(column_list=list(removed_categorical_features))
        self.logger.info("Feature Processing: dropped categorical features")
        # Select high variance features
        data_processor.selecting_high_variance_gene_expression(self.quantile_percentage)
        self.features = data_processor.dataframe.columns
        
        # Fit StandardScaler on numerical columns
        self.scaler = StandardScaler()
        numerical_cols = data_processor.find_cols_on_type('float64')
        self.scaler.fit(data_processor.dataframe[numerical_cols])  
        
        
        # Fit OneHotEncoder on categorical columns
        self.one_hot_encoder = DummyEncoder()
        self.categorizer = Categorizer()
        self.categorizer.fit(data_processor.dataframe)
        self.one_hot_encoder.fit(data_processor.dataframe)
        self.processed_df = data_processor.dataframe

        return self


    def transform(self, X):

        if not isinstance(X, dd.DataFrame):
            X = dd.from_pandas(X, npartitions=4)  
        
        data_processor = DaskDataProcessor(X)
        data_processor.dataframe = data_processor.dataframe[self.features]
        data_processor.dataframe = data_processor.dataframe.persist()
        # Apply StandardScaler to numerical columns
        numerical_cols = data_processor.find_cols_on_type('float64')
        data_processor.dataframe[numerical_cols] = self.scaler.transform(data_processor.dataframe[numerical_cols])
        self.logger.info("Feature Processing: normalized")
        
        # Apply OneHotEncoder to categorical columns
        data_processor.dataframe = self.categorizer.transform(data_processor.dataframe)
        data_processor.dataframe = self.one_hot_encoder.transform(data_processor.dataframe)
        self.logger.info("Feature Processing: one hot encoded")
        # Fill NaN values with 0
        data_processor.dataframe = data_processor.dataframe.map_partitions(lambda x: x.fillna(0))
        self.logger.info("Feature Processing: nan values replaced by 0")

        return data_processor.dataframe