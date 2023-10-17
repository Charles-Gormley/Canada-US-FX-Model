# Python Modules
import logging
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)

# Standard Data Science Libraries
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, log_loss


pickle_file = 'appml-assignment1-dataset-v2.pkl'
df = pd.read_pickle(pickle_file)

df_x = pd.DataFrame(df['X'])
df_y = pd.Series(df['y'])


# a) Fill in missing values & standardize numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# b) Process date feature - this will require custom transformers
from sklearn.base import BaseEstimator, TransformerMixin

class CustomDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col_name='date'):
        logging.info("Function Custom Date Generator init() has been called")
        self.date_col_name = date_col_name

    def fit(self, X, y=None):
        logging.info("Function Custom Date Generator fit() has been called")
        return self

    def transform(self, X):
        X_ = X.copy()
        logging.info("Function Custom Date Generator transform() has been called")
        X_[self.date_col_name] = pd.to_datetime(X_[self.date_col_name])
        X_['day'] = X_[self.date_col_name].dt.day
        X_['hour'] = X_[self.date_col_name].dt.hour
        X_['quarter'] = X_[self.date_col_name].dt.quarter
        X_['m'] = X_[self.date_col_name].dt.month
        X_['minute'] = X_[self.date_col_name].dt.minute
        X_['y'] = X_[self.date_col_name].dt.year
        return X_.drop(columns=[self.date_col_name])
    
class CustomeNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, imputer):
        logging.info("Function Custom Numeric Generator init() has been called")
        self.imputer = imputer
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        logging.info("Function Custom Numeric Generator fit() has been called")
        self.imputer.fit(X)
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_ = X.copy()
        logging.info("Function Custom Numeric Generator transform() has been called")

        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        X_df = pd.DataFrame(X, columns=X_.columns)

        return X_df
        

        

# c) One-hot encoding
categorical_transformer = OneHotEncoder()
ordinal_transformer = OrdinalEncoder()

independent_vars = df_x.columns.to_list()
raw_date_column = ['date']
independent_vars.remove('date')
onehot_categorical_features = ['day', 'hour', 'quarter', 'm']
ordinal_categorical_features = ['minute', 'y']

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_features),
#         ('date', DateTransformer(), ['date']),
#         ('one_cat', categorical_transformer, onehot_categorical_features),
#         ('ordcat', ordinal_transformer, ordinal_categorical_features)
#     ])


model_groups = []

# Parameters
feature_sets = [(independent_vars, "whole")] # List of list for columns to keep in this training set.
null_strategies = [
    (SimpleImputer(strategy='mean'), "Mean Imputer"),
    (SimpleImputer(strategy='median'), "Median Imputer"),
    (KNNImputer(n_neighbors=5), "KNN Imputer (k=5)")  # You can adjust n_neighbors as needed
]
models = [
    (LogisticRegression(n_jobs=-1), "Logistic Regression"),
    (LinearRegression(n_jobs=-1), "Linear Regression"),
    (GradientBoostingRegressor(), "Gradient Boosting Regressor"),
    (RandomForestRegressor(), "Random Forest Regressor")
]


raw_date_column = ['date']
onehot_categorical_features = ['day', 'hour', 'quarter', 'm']
ordinal_categorical_features = ['minute', 'y']
scoring = {
    'mse': make_scorer(mean_squared_error),
}
for model in models:
    for ind_vars in feature_sets:
        for null_strategy in null_strategies:
            logging.info(f"Starting the model training of {ind_vars[1]}")
            model_results = dict()
            model_results['model'] = model[1]
            model_results['ind_vars'] = ind_vars[1]
            model_results['imputer'] = null_strategy[1]            
            # Train-Test Split

            numerical_features = ind_vars[0]

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # preprocessor = ColumnTransformer(
            #     transformers=[
            #         ('num', numerical_transformer, numerical_features),
            #         ('cat', categorical_transformer, categorical_cols)
            #     ])


            numerical_transformer = Pipeline(steps=[
                                                ('imputer', SimpleImputer(strategy='mean')),
                                                ('scaler', StandardScaler())
                                            ],
                                            verbose=True
            )    

            cat_processor = ColumnTransformer(transformers=[('date', CustomDateTransformer(), ['date'])])

            preprocessor = ColumnTransformer(transformers=[
                    ('one_cat', categorical_transformer, onehot_categorical_features),
                    ('ordcat', ordinal_transformer, ordinal_categorical_features)
                ]
            )
            num_preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features)])

            # Model Parameters
            cur_pipeline = Pipeline(steps=[
                                           ('cat_processor', cat_processor),
                                           ('preprocessor', preprocessor),
                                           ('num_preprocessor', num_preprocessor),
                                           ('regressor', model[0])],
                                    verbose=True
                                    )
            
            cur_pipeline.fit(df_x, df_y)
            
            # # TODO: Process the entire pipeline as a numpy array? 
            # cv_results = cross_validate(cur_pipeline, df_x, df_y, cv=10, scoring=scoring)
            # mse_scores = cv_results['test_mse']
            # avg_mse = mse_scores.mean()

            # r2_scores = cv_results['test_r2']  # Get the R^2 scores
            # avg_r2 = r2_scores.mean()  # Calculate the average R^2

            # print(f"Model: {model[1]}, Imputation: {null_strategy[1]}, Avg CV Score: {avg_mse}")

            # model_results = avg_mse
            # model_results = avg_r2