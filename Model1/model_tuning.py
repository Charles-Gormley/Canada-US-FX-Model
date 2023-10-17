# Python Modules
import logging

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

class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X, y=None):
        df_x = X.copy()

        df_x['day'] = df_x['date'].dt.dayofweek
        df_x['m'] = df_x['date'].dt.month
        df_x['y'] = df_x['date'].dt.year
        df_x['quarter'] = df_x['date'].dt.quarter
        df_x['hour'] = df_x['date'].dt.hour
        df_x['minute'] = df_x['date'].dt.minute

        df_x = df_x.drop(columns=['date'])
        return df_x

# c) One-hot encoding
categorical_transformer = OneHotEncoder()
ordinal_transformer = OrdinalEncoder()

independent_vars = df_x.columns.to_list()
print(independent_vars)
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
            model_results = dict()
            model_results['model'] = model[1]
            model_results['ind_vars'] = ind_vars[1]
            model_results[''] = null_strategy[1]            
            # model_results[''] = [1]
            # model_results[''] = [1]
            # Train-Test Split


            numerical_transformer = Pipeline(steps=[
                ('imputer', null_strategy),
                ('scaler', StandardScaler())
            ])
        
            
            numerical_features = ind_vars[0]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('date', DateTransformer(), ['date']),
                    ('one_cat', categorical_transformer, onehot_categorical_features),
                    ('ordcat', ordinal_transformer, ordinal_categorical_features)
                ]
            )

            # Model Parameters
            cur_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('regressor', model[0])])
            
            cv_results = cross_validate(cur_pipeline, df_x, df_y, cv=10, scoring=scoring)
            mse_scores = cv_results['test_mse']
            avg_mse = mse_scores.mean()

            r2_scores = cv_results['test_r2']  # Get the R^2 scores
            avg_r2 = r2_scores.mean()  # Calculate the average R^2

            print(f"Model: {model[1]}, Imputation: {null_strategy[1]}, Avg CV Score: {avg_mse}")

            model_results = avg_mse
            model_results = avg_r2