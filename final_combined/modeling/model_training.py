# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Go up two levels from current working directory
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(project_root)

# import my own transformer
from final_combined import (  # noqa: E402
    MyLogTransformer,
    MySimpleImputer,
    MyStandardScaler,
    evaluate_predictions,
    load_data,
)

# %%

# Read prepared data
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data = load_data(base_dir)

data.head(2)

# %%
# Perform train-test-split
X = data.drop(columns=["Time_taken(min)"])
y = data["Time_taken(min)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# %%
# Build GLM pipeline

# Perform different transformer for different set of features
# Based on findings in eda_cleaning notbeook
glm_numerical1_features = ["Delivery_person_Age", "Delivery_person_Ratings"]
glm_numerical2_features = ["Delivery_Distance"]
glm_categorical1_features = [
    "Is_Weekday",
    "Festival",
    "Road_traffic_density",
    "Vehicle_condition",
    "multiple_deliveries",
]
glm_categorical2_features = [
    "Pickup_Day_Bin",
    "Weatherconditions",
    "Type_of_order",
    "Type_of_vehicle",
    "City",
]


# impute median + standscaler
glm_numerical1_transformer = Pipeline(
    steps=[
        ("imputer", MySimpleImputer(strategy="median")),
        ("scaler", MyStandardScaler()),
    ]
)

# impute median + logtransform + standardscaler
glm_numerical2_transformer = Pipeline(
    steps=[
        ("imputer", MySimpleImputer(strategy="median")),
        ("log", MyLogTransformer()),
        ("scaler", MyStandardScaler()),
    ]
)

# impute mode
glm_categorical1_transformer = Pipeline(
    steps=[("imputer", MySimpleImputer(strategy="mode"))]
)

# impute mode + onehot encoder
glm_categorical2_transformer = Pipeline(
    steps=[
        ("imputer", MySimpleImputer(strategy="mode")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# combine into a ColumnTransformer
glm_preprocessor = ColumnTransformer(
    transformers=[
        ("num1", glm_numerical1_transformer, glm_numerical1_features),
        ("num2", glm_numerical2_transformer, glm_numerical2_features),
        ("cat1", glm_categorical1_transformer, glm_categorical1_features),
        ("cat2", glm_categorical2_transformer, glm_categorical2_features),
    ]
)

# build GLM pipeline
glm_pipeline = Pipeline(
    steps=[("preprocessor", glm_preprocessor), ("model", GammaRegressor())]
)

# fit + predict
glm_pipeline.fit(X_train, y_train)
glm_preds = glm_pipeline.predict(X_test)

# evaluate performance for this initial model as base line
print(f"Initial GLM R2 Score (as basline): {r2_score(y_test, glm_preds)}")
print(f"Initial GLM MSE (as basline): {mean_squared_error(y_test, glm_preds)}")

# print the hyperparameters of this baseline model
glm_model = glm_pipeline.named_steps["model"]
print("Default Hyperparameters for GammaRegressor:")
print(glm_model.get_params())

# evaluate performance
glm_default_results_df = evaluate_predictions(
    df=pd.DataFrame({"actual": y_test, "predicted": glm_preds}),
    outcome_column="actual",
    preds_column="predicted",
)

print(f"GLM (untuned) Evaluation Summary:")
print(glm_default_results_df)

# %%
# cross validation for GLM Gaussian

# param grid for tuning alpha
param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    glm_pipeline,
    param_grid=param_grid,
    scoring="r2",  # use r2_score as main evaluation metric
    cv=5,  # 5-fold cross-validation
    verbose=0,
    n_jobs=-1,
)

# fit the grid search on the training data
grid_search.fit(X_train, y_train)

# get the best parameters and score
glm_tuned_preds = grid_search.best_estimator_.predict(X_test)

print(f"GridSearch-Tuned GLM Gamma R2 Score: {r2_score(y_test, glm_tuned_preds)}")
print(f"GridSearch-Tuned GLM Gamma MSE: {mean_squared_error(y_test, glm_tuned_preds)}")

glm_tuned_model = grid_search.best_estimator_.named_steps["model"]
print("Tuned Hyperparameters for GammaRegressor:")
print(glm_tuned_model.get_params())

glm_best_params = grid_search.best_params_
glm_best_r2 = grid_search.best_score_
print(f"Best Parameters: {glm_best_params}")
print(f"Best Cross-Validated R²: {glm_best_r2}")

# evaluate performance
glm_results_df = evaluate_predictions(
    df=pd.DataFrame({"actual": y_test, "predicted": glm_tuned_preds}),
    outcome_column="actual",
    preds_column="predicted",
)

# Display evaluation results
print(f"GLM (tuned) Evaluation Summary:")
print(glm_results_df)
# %%

# build LGBM pipeline

# feature groups
# LGBM do not need standardscaler
lgbm_numerical_features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Delivery_Distance",
]
lgbm_categorical1_features = [
    "Is_Weekday",
    "Festival",
    "Road_traffic_density",
    "Vehicle_condition",
    "multiple_deliveries",
]
lgbm_categorical2_features = [
    "Pickup_Day_Bin",
    "Weatherconditions",
    "Type_of_order",
    "Type_of_vehicle",
    "City",
]


# impute median
lgbm_numerical_transformer = Pipeline(
    steps=[("imputer", MySimpleImputer(strategy="median"))]
)


# impute mode
lgbm_categorical1_transformer = Pipeline(
    steps=[("imputer", MySimpleImputer(strategy="mode"))]
)

# impute mode + LabelEncoder
lgbm_categorical2_transformer = Pipeline(
    steps=[("imputer", MySimpleImputer(strategy="mode")), ("encoder", OrdinalEncoder())]
)


# combine into ColumnTransformer
lgbm_preprocessor = ColumnTransformer(
    transformers=[
        ("num", lgbm_numerical_transformer, lgbm_numerical_features),
        ("cat1", lgbm_categorical1_transformer, lgbm_categorical1_features),
        ("cat2", lgbm_categorical2_transformer, lgbm_categorical2_features),
    ]
)

# build LightGBM pipeline
lgbm_pipeline = Pipeline(
    steps=[
        ("preprocessor", lgbm_preprocessor),
        (
            "model",
            LGBMRegressor(
                objective="gamma", n_estimators=1000, learning_rate=0.1, num_leaves=6
            ),
        ),  # use gamma distribution for consistency with GLM
    ]
)

# set hyperparameter grid
param_grid = {
    "model__learning_rate": [0.01, 0.05, 0.1],  # learning_rate
    "model__num_leaves": [20, 31, 50],  # num_leaves
    "model__min_child_weight": [1, 3, 5],  # min_child_weight
}

# conduct GridSearchCV
grid_search_lgbm = GridSearchCV(
    estimator=lgbm_pipeline,
    param_grid=param_grid,
    scoring="r2",  # Use R² as the evaluation metric
    cv=5,  # 5-fold cross-validation
    verbose=0,
    n_jobs=-1,
)

# fit GridSearchCV
grid_search_lgbm.fit(X_train, y_train)

# get best parameters and best cross-validated R²
best_params = grid_search_lgbm.best_params_
best_r2 = grid_search_lgbm.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validated R²: {best_r2}")

lgbm_tuned_model = grid_search_lgbm.best_estimator_
lgbm_tuned_preds = lgbm_tuned_model.predict(X_test)

# evaluate performance on the test set
lgbm_best_r2 = r2_score(y_test, lgbm_tuned_preds)
lgbm_best_mse = mean_squared_error(y_test, lgbm_tuned_preds)
print(f"LGBM tuned R2 Score: {lgbm_best_r2}")
print(f"LGBM tuned MSE: {lgbm_best_mse}")

print("Tuned Hyperparameters for LGBMRegressor:")
print(lgbm_tuned_model.named_steps["model"].get_params())

# %%
lgbm_results_df = evaluate_predictions(
    df=pd.DataFrame({"actual": y_test, "predicted": lgbm_tuned_preds}),
    outcome_column="actual",
    preds_column="predicted",
)

lgbm_results_df
# %%
# compare prediction results

plt.figure(figsize=(12, 10))
plt.scatter(glm_tuned_preds, y_test, alpha=0.6, label="GLM Predictions", color="blue")
plt.scatter(
    lgbm_tuned_preds, y_test, alpha=0.6, label="LGBM Predictions", color="green"
)

# add the diagonal line
min_val = min(np.min(y_test), np.min(glm_tuned_preds), np.min(lgbm_tuned_preds))
max_val = max(np.max(y_test), np.max(glm_tuned_preds), np.max(lgbm_tuned_preds))
plt.plot([min_val, max_val], [min_val, max_val], "--r", label="Perfect Prediction")

plt.xlabel("Predicted Values", fontsize=12)
plt.ylabel("Actual Values", fontsize=12)
plt.title("Predicted vs. Actual: GLM vs. LGBM", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%

# compare evaluation results
glm_default_results_df.columns = ['GLM_Default']
glm_results_df.columns = ['GLM_Tuned']
lgbm_results_df.columns = ['LGBM_Tuned']
combined_results_df = pd.concat(
    [glm_default_results_df, glm_results_df, lgbm_results_df],
    axis=1
)


print(combined_results_df)

# %%

# find relevant features by sorting coefficients for GLM
glm_tuned_coef = glm_tuned_model.coef_

# feature names after the pipeline transformtion
glm_feature_names = (
    glm_numerical1_features
    + glm_numerical2_features
    + glm_categorical1_features
    + list(
        glm_preprocessor.named_transformers_["cat2"]
        .named_steps["onehot"]
        .get_feature_names_out(glm_categorical2_features)
    )  # Get one-hot encoded feature names
)

glm_feature_importance = pd.DataFrame(
    {"Feature": glm_feature_names, "Coefficient": glm_tuned_coef}
).sort_values(
    by="Coefficient", key=abs, ascending=False
)  # Sort by absolute value

print(glm_feature_importance)
# %%

lgbm_tuned_model0 = lgbm_tuned_model.named_steps["model"]
lgbm_importance = lgbm_tuned_model0.feature_importances_

# feature names for LGBM
lgbm_feature_names = (
    lgbm_numerical_features + lgbm_categorical1_features + lgbm_categorical2_features
)

lgbm_feature_importance = pd.DataFrame(
    {"Feature": lgbm_feature_names, "Importance": lgbm_importance}
).sort_values(by="Importance", ascending=False)

print(lgbm_feature_importance)


# %%
# top 5 important features from lgbm
top_features = list(lgbm_feature_importance.head(5)["Feature"])

preprocessor = lgbm_tuned_model.named_steps["preprocessor"]

# get the X_test after LGBM pipeline
X_test_transformed = pd.DataFrame(
    preprocessor.transform(X_test), columns=lgbm_feature_names
)

# partial dependence plots for the top 5 features
PartialDependenceDisplay.from_estimator(
    estimator=lgbm_tuned_model0,
    X=X_test_transformed,
    features=top_features,
    feature_names=lgbm_feature_names,
    kind="average",
    grid_resolution=50,
    ax=plt.subplots(1, 1, figsize=(10, 6))[1],
)

# Show the plot
plt.show()

# %%
