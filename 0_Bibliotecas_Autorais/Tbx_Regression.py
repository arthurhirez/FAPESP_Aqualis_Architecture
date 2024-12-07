import pandas as pd
pd.set_option('display.max_columns', None)

import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from Tbx_Processing import create_labels

from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime


import numpy as np


import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    )
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor



import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd

from datetime import datetime
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from scipy import stats





def preprocessing_pipeline(
        data_raw, target, only_split = False,
        ordinal_columns = None, ordinal_orderings = None,
        nominal_columns = None, numeric_columns = None,
        test_size = 0.2, random_state = 42, scaler_type = 'standard',
        stratified_kfold = None, n_splits = 5, shuffle = True):
    """
    Preprocess data with an optional stratified k-fold cross-validation split.

    Parameters:
    - data_raw (pd.DataFrame): The input dataframe.
    - target (str): Target column name.
    - ordinal_columns (list): List of ordinal columns to encode.
    - ordinal_orderings (dict): A dictionary mapping ordinal columns to lists specifying the order.
    - nominal_columns (list): List of nominal columns to one-hot encode.
    - numeric_columns (list): List of numeric columns to scale.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.
    - scaler_type (str): Type of scaler to use ('standard' or 'minmax').
    - stratified_kfold (bool or None): If True, use stratified k-fold split; if None, split into train/test.
    - n_splits (int): Number of splits for k-fold.
    - shuffle (bool): Whether to shuffle data before splitting.

    Returns:
    - If stratified_kfold is None: X_train, X_test, y_train, y_test.
    - If stratified_kfold is True: List of (X_train_fold, X_valid_fold, y_train_fold, y_valid_fold) for each fold.
    """

    # Separate features and target variable
    X = data_raw.drop(columns = [target])
    y = np.array(data_raw[target])

    # Preprocess numeric, ordinal, and nominal columns
    transformers = []

    # Set up scalers
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    
    if numeric_columns:
        transformers.append(('num', scaler, numeric_columns))

    if ordinal_columns and ordinal_orderings:
        for col in ordinal_columns:
            if col in ordinal_orderings:
                transformers.append(('ord_' + col, OrdinalEncoder(categories = [ordinal_orderings[col]]), [col]))

    if nominal_columns:
        transformers.append(('nom', OneHotEncoder(drop = 'first'), nominal_columns))

    # Combine transformations using ColumnTransformer
    preprocessor = ColumnTransformer(transformers = transformers, remainder = 'passthrough')

    # If stratified k-fold is requested, apply it with the preprocessing pipeline
    if stratified_kfold:
        skf = StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)

        folds = []
        for train_idx, valid_idx in skf.split(X, y):
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            if only_split:
                # Append the processed fold to the list
                folds.append((X_train_fold, X_valid_fold, y_train_fold, y_valid_fold))
            else:
                # Fit the pipeline on the training fold and transform both train and validation folds
                pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
                X_train_fold_processed = pipeline.fit_transform(X_train_fold)
                X_valid_fold_processed = pipeline.transform(X_valid_fold)
    
                # Append the processed fold to the list
                folds.append((X_train_fold_processed, X_valid_fold_processed, y_train_fold, y_valid_fold))

        return folds

    # If stratified_kfold is None, do a standard train-test split
    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size, random_state = random_state, stratify = y if stratified_kfold else None
            )

        if only_split:
            return X_train, X_test, y_train, y_test
            
        # Create a pipeline that will be fitted only on the training data
        pipeline = Pipeline(steps = [('preprocessor', preprocessor)])

        # Fit the pipeline on the training data and transform both train and test sets
        pipeline.fit(X_train)
        columns_names = [name.split('__')[1] for name in pipeline.get_feature_names_out().tolist()]

        X_train_processed = pipeline.transform(X_train)
        X_test_processed = pipeline.transform(X_test)

        X_train_processed = pd.DataFrame(X_train_processed, columns = columns_names)
        X_test_processed = pd.DataFrame(X_test_processed, columns = columns_names)

        return X_train_processed, X_test_processed, y_train, y_test





def regression_crossval_compare(
        X_train, y_train, model_dict = None, n_splits = 5, random_state = 42, shuffle = True):
    """
    Perform cross-validation for multiple regression models and compile comparison results.

    Parameters:
    - X_train (array-like): Training features.
    - y_train (array-like): Training target values.
    - n_splits (int): Number of K-Fold splits.
    - random_state (int): Random state for reproducibility.
    - shuffle (bool): Whether to shuffle the data before splitting into folds.

    Returns:
    - metrics_df (pd.DataFrame): A DataFrame comparing average MAE, MSE, RMSE, and R² across folds for each model.
    """

    # Convert to NumPy arrays if inputs are DataFrame or Series
    if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
        X_train = X_train.copy().to_numpy()
    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
        y_train = y_train.copy().to_numpy()

    if model_dict is None:
        # Dictionary of available models
        model_dict = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state = random_state),
            'Lasso Regression': Lasso(random_state = random_state),
            'Random Forest': RandomForestRegressor(random_state = random_state),
            'Decision Tree': DecisionTreeRegressor(random_state = random_state),
            'Support Vector Machine': SVR(),
            'K-Nearest Neighbors': KNeighborsRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(random_state = random_state),
            'XGBoost': XGBRegressor(random_state = random_state),
            'AdaBoost': AdaBoostRegressor(random_state = random_state),
            'Bagging Regressor': BaggingRegressor(random_state = random_state),
            'Extra Trees': ExtraTreesRegressor(random_state = random_state)
            }

    # Initialize the K-Fold cross-validator
    kf = KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)

    # Dictionary to store metrics for each model
    results = {
        'Model': [],
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'R2': []
        }

    # Iterate over each model in the dictionary
    for model_name, model in model_dict.items():
        # Lists to store metrics for each fold for the current model
        mae_list, mse_list, rmse_list, r2_list = [], [], [], []

        # Cross-validation over each fold
        for train_index, valid_index in kf.split(X_train):
            X_fold_train, X_fold_valid = X_train[train_index], X_train[valid_index]
            y_fold_train, y_fold_valid = y_train[train_index], y_train[valid_index]

            # Fit the model on the training fold
            model.fit(X_fold_train, y_fold_train)

            # Predict on the validation fold
            y_pred = model.predict(X_fold_valid)

            # Calculate metrics for the current fold
            mae_list.append(mean_absolute_error(y_fold_valid, y_pred))
            mse_list.append(mean_squared_error(y_fold_valid, y_pred))
            rmse_list.append(np.sqrt(mean_squared_error(y_fold_valid, y_pred)))
            r2_list.append(r2_score(y_fold_valid, y_pred))

        # Calculate average metrics across folds for the current model
        results['Model'].append(model_name)
        results['MAE'].append(np.mean(mae_list))
        results['MSE'].append(np.mean(mse_list))
        results['RMSE'].append(np.mean(rmse_list))
        results['R2'].append(np.mean(r2_list))

    # Compile metrics into a DataFrame
    metrics_df = pd.DataFrame(results)

    # print("Cross-validation results for each model:")
    # print(metrics_df)

    return metrics_df.sort_values(by = 'MAE')





def feature_selection(X, y, method = 'correlation', top_n = None, random_state = 42, return_data = False):
    """
    Select features based on the specified feature selection method.
    
    Parameters:
    - X (pd.DataFrame): The feature matrix.
    - y (pd.Series or np.array): The target variable.
    - method (str): The feature selection method. Options: 'correlation', 'mutual_info', 'model_based'.
    - top_n (int): The number of top features to return. If None, returns all features sorted by importance.
    - random_state (int): Random state for reproducibility (used for model-based selection).
    - return_data (bool): If True, returns the dataset with only the selected features.
    
    Returns:
    - selected_features (pd.DataFrame): A DataFrame with features and their scores, ranked by importance.
    - X_selected (pd.DataFrame, optional): The dataset filtered to include only the selected features.
    """

    feature_scores = pd.DataFrame({'Feature': X.columns})

    # Correlation-based selection
    if method == 'correlation':
        # Calculate the absolute correlation between each feature and the target variable
        correlations = X.apply(lambda col: np.corrcoef(col, y)[0, 1])
        feature_scores['Score'] = correlations.abs()
        feature_scores = feature_scores.dropna()  # Drop features with NaN correlation (e.g., constant features)

    # Mutual information-based selection
    elif method == 'mutual_info':
        # Calculate mutual information between each feature and the target variable
        mutual_info = mutual_info_regression(X, y, random_state = random_state)
        feature_scores['Score'] = mutual_info

    # Model-based selection (Random Forest feature importances)
    elif method == 'model_based':
        # Fit a Random Forest model and extract feature importances
        model = RandomForestRegressor(random_state = random_state)
        model.fit(X, y)
        feature_scores['Score'] = model.feature_importances_

    else:
        raise ValueError("Invalid method. Choose from 'correlation', 'mutual_info', 'model_based'.")

    # Sort features by their scores in descending order
    feature_scores = feature_scores.sort_values(by = 'Score', ascending = False).reset_index(drop = True)

    # Select the top_n features if specified
    if top_n is not None:
        feature_scores = feature_scores.head(top_n)

    # Extract the selected feature names
    selected_feature_names = feature_scores['Feature'].tolist()

    # Filter the dataset to include only selected features if return_data is True
    if return_data:
        X_selected = X[selected_feature_names]
        return feature_scores, X_selected
    else:
        return feature_scores



def evaluate_model(model, X, y):
    """
    Evaluate a regression model and print performance metrics.
    """
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    return mae, mse, rmse, r2


def display_regression_formula(model, feature_names):
    """
    Display the regression formula for a given linear model.
    """
    coefficients = model.coef_
    intercept = model.intercept_
    formula = f"y = {intercept:.4f}"
    for coef, name in zip(coefficients, feature_names):
        formula += f" + ({coef:.4f} * {name})"
    print("\nRegression Formula:\n", formula)


# Step 1: Baseline Linear Regression
def baseline_linear_regression(X, y):
    """
    Fit and evaluate a baseline linear regression model.
    """
    print("\n=== Baseline Linear Regression ===")
    model = LinearRegression()
    model.fit(X, y)
    evaluate_model(model, X, y)
    display_regression_formula(model, X.columns)
    return model



def refine_linear_regression(X, y, regularization = 'none', degree = 2, alpha = 1.0, l1_ratio = 0.5):
    """
    Refine the linear regression model with feature engineering and regularization.
    
    Parameters:
    - X: Features (DataFrame or array)
    - y: Target variable (Series or array)
    - regularization: Type of regularization ('none', 'ridge', 'lasso', 'elasticnet')
    - degree: Degree of polynomial features
    - alpha: Regularization strength
    - l1_ratio: The Elastic Net mixing parameter (0 <= l1_ratio <= 1). Only used if regularization='elasticnet'.
    """
    print(f"\n=== Refined Linear Regression with {regularization.capitalize()} Regularization ===")

    # Polynomial Features
    poly = PolynomialFeatures(degree = degree, include_bias = False)
    X_poly = pd.DataFrame(poly.fit_transform(X), columns = poly.get_feature_names_out(X.columns))

    # Scale features
    # scaler = StandardScaler()
    # X_poly_scaled = scaler.fit_transform(X_poly)

    # Choose model based on regularization type
    if regularization == 'ridge':
        model = Ridge(alpha = alpha)
    elif regularization == 'lasso':
        model = Lasso(alpha = alpha, max_iter = 3000)
    elif regularization == 'elasticnet':
        model = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, max_iter = 3000)  # Add ElasticNet with l1_ratio
    else:  # No regularization
        model = LinearRegression()

    # Fit the model
    model.fit(X_poly, y)
    evaluate_model(model, X_poly, y)  # Assuming this is a custom function you have defined
    display_regression_formula(model, X_poly.columns)  # Assuming this is a custom function you have defined
    return model, X_poly




def evaluate_model_performance(model, X_test, y_test, cv_folds = 5, log_transf = None):
    """
    Evaluate a fitted model with KFold cross-validation, generate a regression report,
    and plot residuals and histogram for graphical analysis.

    Parameters:
    - model: Fitted model object (e.g., from sklearn)
    - X_test: Features for testing the model (DataFrame or numpy array)
    - y_test: True target values for testing the model (Series or numpy array)
    - cv_folds: Number of folds for KFold cross-validation (default is 5)
    
    Returns:
    - cv_results: Cross-validation results (mean score and standard deviation)
    - result_df: DataFrame with y_test, y_pred, and residuals
    """
    # 1. Cross-Validation (KFold)
    kf = KFold(n_splits = cv_folds, shuffle = True, random_state = 42)
    cv_scores = cross_val_score(model, X_test, y_test, cv = kf, scoring = 'neg_mean_squared_error')
    cv_results = {'mean_score': -cv_scores.mean(), 'std_score': cv_scores.std()}

    # 2. Regression Report (Metrics)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    regression_report = {
        'Mean Squared Error (MSE)': mse,
        'Mean Absolute Error (MAE)': mae,
        'R^2': r2
        }

    print("=== Regression Report ===")
    for key, value in regression_report.items():
        print(f"{key}: {value:.4f}")

    # 3. Create DataFrame with y_test, y_pred, and residuals
    residuals = y_test - y_pred
    result_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred,
        'resid': residuals
        })
    result_df['MAE'] = mae
    result_df['MSE'] = mse

    # 4. Simplified Residuals Analysis
    # Residuals Analysis
    plt.figure(figsize = (16, 12))

    # Residuals vs Fitted plot
    plt.subplot(2, 2, 1)
    sns.residplot(x = y_pred, y = residuals, lowess = True, line_kws = {'color': 'red'})
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')

    # Histogram of residuals with KDE
    plt.subplot(2, 2, 2)
    sns.histplot(residuals, kde = True, bins = 30, color = 'blue')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution (Histogram + KDE)')

    # Q-Q plot
    plt.subplot(2, 2, 3)
    stats.probplot(residuals, dist = "norm", plot = plt)
    plt.title('Q-Q Plot of Residuals')

    # Scale-Location plot (sqrt of absolute residuals vs fitted values)
    plt.subplot(2, 2, 4)
    sqrt_resid = np.sqrt(np.abs(residuals))
    sns.scatterplot(x = y_pred, y = sqrt_resid, color = 'green')
    plt.xlabel('Fitted Values')
    plt.ylabel('Sqrt(|Residuals|)')
    plt.title('Scale-Location Plot (Sqrt of Residuals)')

    plt.tight_layout()
    plt.show()

    return cv_results, result_df




def compare_relationships(data, feature, error_metric, error_compare):
    """
    Generates three plots to explore the relationship between 'Error' and a given feature:
    - Scatter Plot with Regression Line
    - Box Plot
    - Histogram with KDE
    
    Parameters:
        data (DataFrame): The dataset containing 'Error' and the specified features.
        feature (str): The feature column name to plot against 'Error'.
    
    Returns:
        None
    """
    # Set up the plotting area
    fig, axes = plt.subplots(1, 2, figsize = (18, 5), sharey=True)

    # First plot: Only the regression line across all data points
    sns.regplot(
        x = feature, y = error_metric, data = data, ax = axes[0],
        scatter = False,  # Hide scatter points
        line_kws = {'color': 'red'}
        )
    axes[0].set_title(f'Regression Line for {feature} vs {error_metric}')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel(error_metric)

    # Second plot: Scatter plot of points where error_metric > 0
    positive_data = data[data[error_metric] >= 0]  # Filter for positive values
    sns.scatterplot(
        x = feature, y = error_metric, data = positive_data, ax = axes[0],
        color = 'green', alpha = 0.5
        )
    axes[0].set_title(f'Scatter Plot (y > 0) for {feature} vs {error_metric}')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel(error_metric)

    # Third plot: Scatter plot of points where error_metric < 0
    negative_data = data[data[error_metric] < 0]  # Filter for negative values
    sns.scatterplot(
        x = feature, y = error_metric, data = negative_data, ax = axes[0],
        color = 'purple', alpha = 0.5
        )
    axes[0].set_title(f'Scatter Plot (y < 0) for {feature} vs {error_metric}')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel(error_metric)



    # First plot: Only the regression line across all data points
    sns.regplot(
        x = feature, y = error_compare, data = data, ax = axes[1],
        scatter = False,  # Hide scatter points
        line_kws = {'color': 'red'}
        )
    axes[1].set_title(f'Regression Line for {feature} vs {error_compare}')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel(error_compare)

    # Second plot: Scatter plot of points where error_compare > 0
    positive_data = data[data[error_compare] >= 0]  # Filter for positive values
    sns.scatterplot(
        x = feature, y = error_compare, data = positive_data, ax = axes[1],
        color = 'green', alpha = 0.5
        )
    axes[1].set_title(f'Scatter Plot (y > 0) for {feature} vs {error_compare}')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel(error_compare)

    # Third plot: Scatter plot of points where error_compare < 0
    negative_data = data[data[error_compare] < 0]  # Filter for negative values
    sns.scatterplot(
        x = feature, y = error_compare, data = negative_data, ax = axes[1],
        color = 'purple', alpha = 0.5
        )
    axes[1].set_title(f'Scatter Plot (y < 0) for {feature} vs {error_compare}')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel(error_compare)



    plt.tight_layout()
    plt.show()



def perform_gridsearch(model, param_grid, X_train, y_train, cv=5, scoring=None):
    """
    Perform grid search with cross-validation on a given model and hyperparameter grid.
    
    Parameters:
    - model: A machine learning model (e.g., LinearRegression, RandomForest, etc.).
    - param_grid: A dictionary containing hyperparameters and their corresponding values to be tuned.
    - X_train: Training features.
    - y_train: Training labels.
    - cv: Number of cross-validation folds (default is 5).
    - scoring: A string or callable to evaluate the predictions on the validation set.
      - If None, the default scorer for the estimator is used (e.g., accuracy for classifiers).
    
    Returns:
    - best_model: The model with the best hyperparameters after grid search.
    """
    
    # Initialize GridSearchCV with the scoring parameter
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    
    # Fit grid search on the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best model from grid search
    best_model = grid_search.best_estimator_
    
    # Display the best parameters and score
    print("\nBest Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    
    return best_model




