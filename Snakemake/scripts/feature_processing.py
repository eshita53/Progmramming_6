
from data_processor import DataProcessors
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureProcessing(TransformerMixin, BaseEstimator):

    def __init__(self, quantile_percentage=95, nan_threshold=35, alpha=0.06):

        self.quantile_percentage = quantile_percentage
        self.nan_threshold = nan_threshold
        self.alpha = alpha

    def fit(self, X, y):
        data_process = DataProcessors(X)
        data_process.remove_nonrelated_columns()

        data_process.impute_notavailable_values('characteristics.tag.grade')
        # print(data_process.percantage_missing_values())
        data_process.drop_nan_columns(self.nan_threshold)
        if data_process.dataframe['characteristics.tag.tumor.size.maximumdiameter'].isna().sum() > 0:
            # We have 1 null values in characteristics.tag.tumor.size.maximumdiameter. we filled the null values of maximumdiameter with average because upper value and lower value of that position was too high. Filling it with average makes it more reasonable value than fill it with upper or lower value. As the sample size is
            data_process.dataframe = DataProcessors.imputing_column_with_average(
                data_process.dataframe, 'characteristics.tag.tumor.size.maximumdiameter')
            # 89, we should not delete this rows. If we do so, our sample size will be more small.

        data_process.dataframe.drop(
            columns=['sample.name', 'title', 'CEL.file'], inplace=True)

        data_process.categorical_feat_sel_using_chi2(y, alpha=self.alpha)
        self.significant_col = data_process.significant_features_cat
        data_process.selecting_high_variance_gene_expression(
            quantile_percentage=self.quantile_percentage)

        self.scaler = StandardScaler()
        cols = DataProcessors.find_cols_on_types(
            data_process.dataframe, 'number')
        self.cat_ohe = OneHotEncoder()
        self.cat_ohe.fit(
            data_process.dataframe[data_process.significant_features_cat])
        self.scaler.fit(data_process.dataframe[cols])
        self.features = data_process.dataframe.columns.to_list()
        return self

    def transform(self, X):

        X = X[self.features]
        if len(self.significant_col) > 0:
            X = DataProcessors.encoding_catagorical_features(
                X, self.cat_ohe, self.significant_col)
        if ('characteristics.tag.tumor.size.maximumdiameter' in X.columns) & (X['characteristics.tag.tumor.size.maximumdiameter'].isna().sum() > 0):
            X = DataProcessors.imputing_column_with_average(
                X, 'characteristics.tag.tumor.size.maximumdiameter')
        X_encoded = DataProcessors.transform_standard_scaling(X, self.scaler)
        if X_encoded.isna().sum().sum() == 0:
            return X_encoded
        else:
            raise (ValueError(f"There are nan values in the dataframe"))
