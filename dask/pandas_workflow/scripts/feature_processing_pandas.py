from sklearn.base import TransformerMixin,BaseEstimator
from data_processor import DataProcessor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureProcessing(TransformerMixin, BaseEstimator): 
    def __init__(self, covariance_threshold=0, quantile_percentage=95, nan_threshold =35):
        self.covariance_threshold = covariance_threshold
        self.quantile_percentage = quantile_percentage
        self.nan_threshold= nan_threshold
        
    def fit(self, X, y=None):
        
        data_processor = DataProcessor(X)
        data_processor.remove_non_related_columns()
        data_processor.impute_not_available_values('characteristics.tag.grade')
        data_processor.drop_nan_columns(self.nan_threshold)
        
        data_processor.cramerV(y, self.covariance_threshold)
        self.covarrianced_columns = data_processor.covarrianced_columns
        removed_catagorical_features = set(data_processor.find_cols_on_type('object')) - set(self.covarrianced_columns)
        data_processor.drop_columns(column_list = list(removed_catagorical_features))
        data_processor.selecting_high_variance_gene_expression(self.quantile_percentage)
        self.features = data_processor.dataframe.columns
        
        self.scaler = StandardScaler()
        self.scaler.fit(data_processor.dataframe[data_processor.find_cols_on_type('float64')])
        
        self.one_hot_encoder= OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.one_hot_encoder.fit(data_processor.dataframe[data_processor.covarrianced_columns])
        self.processed_df = data_processor.dataframe
        return self
    
    def transform(self, X):
    
       data_processor = DataProcessor(X)
       data_processor.dataframe = data_processor.dataframe[self.features]
       data_processor.fit_standard_scaling(self.scaler)
       data_processor.encoding_catagorical_features(self.one_hot_encoder, self.covarrianced_columns)
       data_processor.dataframe.fillna(0, inplace=True)
       
       X = data_processor.dataframe

       return X

