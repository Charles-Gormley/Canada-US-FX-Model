# TODO

## EDA
[ ] Date Analysis [Quarter, January Effect, Monday Effect]
[ ] Correllations with other currencies
[ ] Volatility Correllations [High-Low]
[ ] Do we consider the open price in our next prediction? 
[ ] Multi-Collinearity

## Data Cleaning
[ ] K means to remove Null Values
[ ] Mean Squared Error 

## Feature Sets
* I want to use 1 with everything
* Another where I eye-ball multi-collinearity avoidance and high regression coefficients with dependent variable.
* [Advanced] Calculate Multi-collinearity of each variable with other prediction variables. Remove multi-collinear variable (except for one of course)
* [Advanced] Calculate correllations with the dependent variable (Next hour of trading). Keep good predictor variables while avoiding multi-collinearity.

## Feature Engineering
[ ] Normalize/Scale Features
[ ] Pick the most relevant currencies to the Canadian rate [Basically countries which interact with Canada]
[ ] Seasonality Variables [Quarter, January Effect, Monday Effect]
[ ] Trend Variables [Year itself]
[ ] K-means for null values
[ ] Non Linear Functions
[ ] Volatility Feature High and Low Difference

## Training 
[ ] Shuffle the data
[ ] CV-Fold-10 & CV-Fold-5
[ ] Early Stopping
[ ] Hyperparameter Tuning

## Models
[ ] Pick 2 Models which will be really good with sklearn

## Feature Tuning
[ ] Feature Importance
[ ] Playing around with different currency combinations.
