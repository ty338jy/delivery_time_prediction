import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 20) -> None:
    """
    plot a histogram for a given column (need to be continous numerical column)

    Args:
        df (pd.DataFrame): input dataframe
        column (str): the column name for which to plot the histogram
        bins (int): number of bins for the histogram (default is 20)
    """
    # calculate quantiles
    q25, q50, q75 = df[column].quantile([0.25, 0.5, 0.75])

    # create the plot
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=column, kde=True, bins=bins)

    # add vertical lines for quantiles
    plt.axvline(
        q25,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"25th Percentile: {q25: .2f}",
    )
    plt.axvline(
        q50,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Median (50th): {q50: .2f}",
    )
    plt.axvline(
        q75,
        color="blue",
        linestyle="--",
        linewidth=1,
        label=f"75th Percentile: {q75: .2f}",
    )

    # set title and axis label
    plt.title(column, fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_scatter(df: pd.DataFrame, feature: str, target: str) -> None:
    """
    plot a scatter plot for a specified feature
    against the target variable in a dataframe

    Args:
        df (pd.DataFrame): input dataframe
        feature (str): column name of feature to plot on the x-axis
        target (str): column name of target variable to plot on the y-axis
    """

    plt.figure(figsize=(6, 4))

    # create the scatter plot
    sns.scatterplot(data=df, x=feature, y=target, alpha=0.7, edgecolor="k")

    # add a regression line
    sns.regplot(data=df, x=feature, y=target, scatter=False, color="red")

    # set title and labels
    plt.title(feature, fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_coordinates(df, from_lat, from_lon, to_lat, to_lon):
    """
    plot coordinates for two sets of latitude and longitude points

    Args:
        df (pd.DataFrame): input df containing latitude and longitude columns.
        from_lat (str): column name for the first set of latitude values.
        from_lon (str): column name for the first set of longitude values.
        to_lat (str): column name for the second set of latitude values.
        to_lon (str): column name for the second set of longitude values.
    """
    plt.figure(figsize=(6, 4))

    # plot the first set of coordinates
    plt.scatter(
        df[from_lon],
        df[from_lat],
        color="blue",
        marker="o",
        alpha=0.6,
        label="From Locations",
        edgecolor="k",
    )

    # plot the second set of coordinates
    plt.scatter(
        df[to_lon],
        df[to_lat],
        color="red",
        marker="x",
        alpha=0.6,
        label="To Locations",  # noqa :E501
    )

    plt.title("Coordinates Plot", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.5, linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(df: pd.DataFrame, feature: str) -> None:
    """
    Plots a bar chart showing the distribution
    of a categorical variable, including NaN values.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name of the categorical variable.
        title (str, optional): Title of the plot. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (8, 6).

    Returns:
        None: Displays the bar chart.
    """
    # Count values, including NaN
    counts = df[feature].value_counts(dropna=False)

    # Create a bar chart
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")

    # Customize plot
    plt.title(f"Distribution of {feature}", fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_boxplot(df: pd.DataFrame, feature: str, target: str) -> None:
    """
    plot a box plot of a given categorical feature
    and continuous target variable
    Args:
        df (pd.DataFrame): input dataframe
        feature (str): name of the categorical column.
        target (str): name of the continuous target variable.
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=feature, y=target, palette="pastel")
    plt.title(f"{target} by {feature}", fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_heatmap(
    df: pd.DataFrame,
    target: str,
    numerical_features: list,
    categorical_features: list,  # noqa :E501
) -> None:
    """
    Creates a combined heatmap of numerical correlation and target

    Args:
        df (pd.DataFrame): The dataset
        target (str): The target variable
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
    """
    # Numerical Correlations
    numerical_corr = (
        df[numerical_features + [target]].corr()[[target]].drop(index=target)
    )
    numerical_corr.columns = ["Correlation"]

    # Categorical Correlations (using Target Mean Encoding)
    categorical_corr = {}
    for feature in categorical_features:
        target_means = df.groupby(feature)[target].mean()
        encoded_corr = df[feature].map(target_means).corr(df[target])
        categorical_corr[feature] = encoded_corr

    categorical_corr_df = pd.DataFrame.from_dict(
        categorical_corr, orient="index", columns=["Correlation"]
    )

    # Combine Numerical and Categorical Correlations
    combined_corr = pd.concat([numerical_corr, categorical_corr_df], axis=0)
    combined_corr = combined_corr.sort_values(
        by="Correlation", ascending=False
    )  # noqa :E501

    # Plot Heatmap
    plt.figure(figsize=(8, 10))
    sns.heatmap(
        combined_corr,
        annot=True,
        cmap="coolwarm",
        cbar=True,
        fmt=".2f",
        linewidths=0.5,  # noqa :E501
    )
    plt.title(f"Correlation of Features with {target}")
    plt.ylabel("Feature")
    plt.xlabel("Correlation")
    plt.show()
