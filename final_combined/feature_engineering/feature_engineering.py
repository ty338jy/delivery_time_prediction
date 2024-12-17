import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MySimpleImputer(BaseEstimator, TransformerMixin):
    """
    re-implementation of sklearn SimpleImputer
    replace missing values with a specified strategy: mean, median, or mode
    """

    def __init__(self, strategy="mean", fill_value=None):
        """
        Args:
            strategy (str): The imputation strategy
            ('mean', 'median', 'mode', or 'constant')
            fill_value (Any): Value to use when strategy is 'constant'
        """
        valid_strategies = ["mean", "median", "mode", "constant"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}")
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """
        find the imputation values based on the specified strategy

        Args:
            X (pd.DataFrame): input data

        Returns:
            self: fitted transformer
        """
        if self.strategy == "mean":
            self.fill_values_ = X.mean()
        elif self.strategy == "median":
            self.fill_values_ = X.median()
        elif self.strategy == "mode":
            # Compute the mode for each column
            self.fill_values_ = X.mode().iloc[0]
        elif self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("Constant Missing")
            self.fill_values_ = pd.Series(self.fill_value, index=X.columns)
        return self

    def transform(self, X):
        """
        fill in missing values in X using the computed values.

        Args:
            X (pd.DataFrame): input data

        Returns:
            pd.DataFrame: transformed data with missing values imputed
        """
        X = X.copy()

        # Fill missing values and preserve original data types
        for col in X.columns:
            if col in self.fill_values_:
                missing_mask = X[col].isnull()
                X.loc[missing_mask, col] = self.fill_values_[col]
                # Explicitly cast to the original data type of the column
                X[col] = X[col].astype(X[col].dtype, errors="ignore")

        return X


class MyStandardScaler(BaseEstimator, TransformerMixin):
    """
    re-implementation of StandardScaler.
    x_new = (x_old) - mu / std
    """

    def fit(self, X, y=None):
        self.mean_ = X.mean()
        self.std_ = X.std()
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


class MyLogTransformer(BaseEstimator, TransformerMixin):
    """
    conduct log transformation for numerical data
    add a small constant to handle zeros and negative values.
    """

    def __init__(self, offset=1e-6):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X + self.offset)
