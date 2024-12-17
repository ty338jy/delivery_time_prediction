import numpy as np
import pandas as pd
import pytest

from final_combined.feature_engineering.feature_engineering import (
    MyLogTransformer,
    MySimpleImputer,
    MyStandardScaler,
)


# Test
@pytest.mark.parametrize(
    "strategy, fill_value, input_data, expected",
    [
        ("mean", None, {"A": [4.8, 4.9, None, 5]}, [4.8, 4.9, 4.9, 5]),  # noqa: E501
        (
            "median",
            None,
            {"A": [36.0, 37.0, None, 39.0]},
            [36.0, 37.0, 37.0, 39.0],
        ),  # test for median
        (
            "mode",
            None,
            {"A": ["High", "High", "Low", None]},
            ["High", "High", "Low", "High"],
        ),  # test for mode
        ("constant", 0, {"A": [1, None, 2]}, [1.0, 0, 2.0]),  # noqa: E501
    ],
)
def test_my_simple_imputer(strategy, fill_value, input_data, expected):
    df = pd.DataFrame(input_data)
    myimputer = MySimpleImputer(strategy=strategy, fill_value=fill_value)
    myimputer.fit(df)
    transformed = myimputer.transform(df)
    pd.testing.assert_frame_equal(
        transformed,
        pd.DataFrame({"A": expected}),
        check_exact=False,
        atol=1e-6,  # noqa: E501
    )


def test_my_simple_imputer_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid strategy: invalid_strategy"):
        MySimpleImputer(strategy="invalid_strategy")


def test_my_simple_imputer_constant_missing_fill_value():
    with pytest.raises(ValueError, match="Constant Missing"):
        MySimpleImputer(strategy="constant").fit(
            pd.DataFrame({"A": [None, 2, 3]})
        )  # noqa: E501


# %%
# Test
@pytest.mark.parametrize(
    "input_data, expected_mean, expected_std, expected_transformed",
    [
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [2.0, 5.0],
            [1.0, 1.0],  # using sample std
            {"A": [-1.0, 0.0, 1.0], "B": [-1.0, 0.0, 1.0]},
        ),
    ],
)
def test_my_standard_scaler(
    input_data, expected_mean, expected_std, expected_transformed
):
    """
    Test MyStandardScaler using multiple test cases with parameterized inputs.
    """
    df = pd.DataFrame(input_data)
    myscaler = MyStandardScaler()

    # Fit and check mean/std
    myscaler.fit(df)
    np.testing.assert_array_almost_equal(
        myscaler.mean_.values, expected_mean, decimal=6
    )
    np.testing.assert_array_almost_equal(
        myscaler.std_.values, expected_std, decimal=6
    )  # noqa: E501

    # Transform and validate
    transformed = myscaler.transform(df)
    # Validate transformed data
    expected_transformed_df = pd.DataFrame(expected_transformed)
    pd.testing.assert_frame_equal(
        pd.DataFrame(transformed, columns=df.columns),
        expected_transformed_df,
        check_exact=False,
        atol=1e-6,
    )


# Tests for MyLogTransformer
@pytest.mark.parametrize(
    "input_data, offset, expected",
    [
        # Basic log transformation
        (
            {"A": [1, 10, 100]},
            1e-6,
            {"A": [np.log(1 + 1e-6), np.log(10 + 1e-6), np.log(100 + 1e-6)]},
        ),
        # Log transformation with offset
        (
            {"A": [0, 1, 10]},
            1,
            {"A": [np.log(0 + 1), np.log(1 + 1), np.log(10 + 1)]},
        ),  # noqa: E501
        # Negative values with offset
        (
            {"A": [-1, 0, 10]},
            2,
            {"A": [np.log(-1 + 2), np.log(0 + 2), np.log(10 + 2)]},
        ),  # noqa: E501
    ],
)
def test_my_log_transformer(input_data, offset, expected):
    """
    Test MyLogTransformer with different offsets and inputs.
    """
    df = pd.DataFrame(input_data)
    transformer = MyLogTransformer(offset=offset)
    transformed = transformer.transform(df)
    expected_df = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(
        transformed, expected_df, check_exact=False, atol=1e-6
    )


# %%
