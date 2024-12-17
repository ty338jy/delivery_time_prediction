import math
from datetime import time

import numpy as np
import pandas as pd


def clean_column_with_regex(
    df: pd.DataFrame, column: str, pattern: str
) -> pd.DataFrame:
    """
    clean a specified column in the DataFrame
    by extracting values based on a regex pattern

    Args:
        df (pd.DataFrame): input dataframe
        column (str):  name of the column to clean
        pattern (str): the regex pattern to use for extracting values

    Returns:
        pd.DataFrame: The DataFrame with the cleaned column.
    """

    # Apply regex extraction to the specified column
    df[column] = df[column].str.extract(pattern).astype(float)
    return df


def convert_to_numerical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    convert string representations of numerical values
    in specified columns into float

    Args:
        df (pd.DataFrame): input dataframe
        columns (list): list of column names

    Returns:
        pd.DataFrame:
        df with specified columns converted to numerical types
    """

    def try_convert(value):
        """
        convert a string value to a float
        if conversion fails (NaN), returns the original value
        """
        if isinstance(value, str):
            value = value.strip()  # Remove extra whitespace
            try:
                return float(value)
            except ValueError:
                return value  # Return original value if not NaN
        return value

    # Apply conversion only to specified columns
    for col in columns:
        df[col] = df[col].apply(try_convert)
    return df


def compute_haversine_distance(
    from_lat: float, from_lon: float, to_lat: float, to_lon: float
) -> float:
    """
    compute the Haversine distance between two points on the Earth.

    Args:
        from_lat (float): Latitude of the first location (restaurant).
        from_lon (float): Longitude of the first location (restaurant).
        to_lat (float): Latitude of the second location (delivery location).
        to_lon (float): Longitude of the second location (delivery location).

    Returns:
        float: Distance between the two points in kilometers.
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    from_lat_rad = math.radians(from_lat)
    from_lon_rad = math.radians(from_lon)
    to_lat_rad = math.radians(to_lat)
    to_lon_rad = math.radians(to_lon)

    # Differences in coordinates
    delta_lat = to_lat_rad - from_lat_rad
    delta_lon = to_lon_rad - from_lon_rad

    # Haversine formula
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(from_lat_rad)
        * math.cos(to_lat_rad)
        * math.sin(delta_lon / 2) ** 2  # noqa: E501
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance
    distance = R * c
    return distance


def is_weekday(orderdate: pd.Timestamp) -> int:
    """
    determine whether a given date is a weekday

    Args:
        orderdate (pd.Timestamp): input date

    Returns:
        int: Returns 1 if the date is a weekday (Monday-Friday),
            0 if it's a weekend (Saturday-Sunday).
            -1 if the input is None or invalid.
    """
    if pd.isnull(orderdate):
        return -1  # Indicate invalid date

    # Get the day of the week (0=Monday, 6=Sunday)
    day_of_week = orderdate.weekday()

    if day_of_week in [5, 6]:  # Saturday and Sunday
        return 0
    else:  # Weekday
        return 1


def calculate_time_difference(pickup: time, order: time) -> float:
    """
    Calculate the difference between two datetime.time objects in minutes
    If either

    Args:
        pickup (datetime.time):
        the order pick-up time (Time_Order_picked column)
        order (datetime.time): the order time (Time_Orderd column)

    Returns:
        float: The difference in seconds.
    """
    if pd.isnull(pickup) or pd.isnull(order):
        return np.nan
    # Convert time objects to seconds since midnight
    pickup_second = pickup.hour * 3600 + pickup.minute * 60 + pickup.second
    order_second = order.hour * 3600 + order.minute * 60 + order.second

    # Calculate the difference
    difference = pickup_second - order_second

    # Handle midnight crossover
    if difference < 0:
        difference += 24 * 3600  # Add 24 hours in seconds

    return difference / 60


def create_time_bin(pickup: time) -> str:
    """
    Categorizes the time of day based on a datetime.time object.

    Args:
        picked_time (datetime.time): The time of day (a datetime.time object).

    Returns:
        str: The time-of-day category
        ('Morning', 'Afternoon', 'Evening', 'Night').
             Returns 'Invalid Time' if the input is None.
    """
    if pickup is None:
        return "Invalid Time"  # Handle None values gracefully

    hour = pickup.hour  # Extract the hour from the time object

    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"
