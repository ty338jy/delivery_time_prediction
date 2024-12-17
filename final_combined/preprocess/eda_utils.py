import numpy as np
import pandas as pd


def standardize_missing_values(df: pd.DataFrame, placeholders=None):
    """
    handle missing values in a given dataframe by
    replacing other string representations of missing values with np.nan

    Args:
        df (pd.DataFrame): input dataframe
        placeholders (list, optional): Additional placeholder values

    Returns:
        pd.DataFrame: dataframe with missing values standardized to np.nan
        dict: dict with keys as column and values as placeholders
    """
    # default placeholders for missing values
    default_placeholders = [
        "nan",
        "none",
        "",
        "null",
        "?",
        "-",
        "--",
        "n/a",
        "na",
        "undefined",
        "conditions nan",
    ]

    # if additional placeholder values are used
    if placeholders:
        all_placeholders = set(default_placeholders + placeholders)
    else:
        all_placeholders = set(default_placeholders)

    # dictionary to track replaced values
    replaced_values = {}

    # iterate to replace values and track replaced placeholders
    df_cleaned = df.copy()
    for col in df.columns:
        replaced_in_col = set()
        cleaned_col = []
        for value in df[col]:
            if (
                isinstance(value, str)
                and value.strip().lower() in all_placeholders  # noqa: E501
            ):  # noqa: E501
                replaced_in_col.add(value)
                cleaned_col.append(np.nan)
            else:
                cleaned_col.append(value)
        df_cleaned[col] = cleaned_col

        # add to dictionary of values have been replaced
        if replaced_in_col:
            replaced_values[col] = replaced_in_col

    return df_cleaned, replaced_values


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    report the missing values in a given dataframe, show counts and percentages

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: a dataframe summarizing missing value information
    """
    # Calculate missing values metrics
    missing_report = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": df.isnull().sum(),
            "Percentage Missing": df.isnull().mean() * 100,
        }
    )
    missing_report = missing_report.sort_values(
        by="Missing Count", ascending=False
    )  # noqa: E501

    return missing_report.reset_index(drop=True)


def report_column_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    report a descriptive statistics summary of all columns in a dataframe
    This function reports various statistics for each column including:
        - Data type
        - Unique value count
        - Missing value count and percentage
        - Mode (most frequent value)
        - Minimum, maximum, mean, median, and standard deviation


    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - Column: The name of the column.
            - Data Type: The data type of the column.
            - UniqueCount: The number of unique values in the column.
            - MissingCount: The number of missing values in the column.
            - MissingPct: The percentage of missing values in the column.
            - Mode: The most frequent value in the column (if applicable).
            - Minimum: The minimum value in the column (for numeric columns).
            - Maximum: The maximum value in the column (for numeric columns).
            - Mean: The mean value of the column (for numeric columns).
            - Median: The median value of the column (for numeric columns).
            - Std: The standard deviation of the column (for numeric columns).

    """
    stats = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [dtype for dtype in df.dtypes],
            "UniqueCount": [df[col].nunique() for col in df.columns],
            "MissingCount": [df[col].isnull().sum() for col in df.columns],
            "MissingPct": [
                df[col].isnull().mean() * 100 for col in df.columns
            ],  # noqa: E501
            "Mode": [
                df[col].mode().iloc[0]
                if not df[col].mode().empty
                else None  # noqa: E501
                for col in df.columns
            ],
            "Minimum": [
                df[col].min()
                if np.issubdtype(df[col].dtype, np.number)
                else None  # noqa: E501
                for col in df.columns
            ],
            "Maximum": [
                df[col].max()
                if np.issubdtype(df[col].dtype, np.number)
                else None  # noqa: E501
                for col in df.columns
            ],
            "Mean": [
                df[col].mean()
                if np.issubdtype(df[col].dtype, np.number)
                else None  # noqa: E501
                for col in df.columns
            ],
            "Median": [
                df[col].median()
                if np.issubdtype(df[col].dtype, np.number)
                else None  # noqa: E501
                for col in df.columns
            ],
            "Std": [
                df[col].std()
                if np.issubdtype(df[col].dtype, np.number)
                else None  # noqa: E501
                for col in df.columns
            ],
        }
    )
    return stats


def categorical_summary(
    df: pd.DataFrame, feature: str, target: str
) -> pd.DataFrame:  # noqa: E501
    """
    compute summary statistics (median, IQR, min, max, mean)
    for a continuous target variable grouped by a categorical variable

    Args:
        df (pd.DataFrame): input dataframe
        categorical_col (str): The categorical variable column name
        target_col (str): The target variable column name

    Returns:
        pd.DataFrame: Summary statistics grouped by the categorical variable
    """
    summary = df.groupby(feature)[target].agg(
        Median="median",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75),
        Min="min",
        Max="max",
        Mean="mean",
    )
    return summary.reset_index()
