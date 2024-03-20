from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle as pkl
import numpy as np
import torch

with open("../data/processed/traj_and_pupil_data.pkl", "rb") as f:
    data = pkl.load(f)

X = np.concatenate(data["trajectories"], axis=1).T
y = np.concatenate(data["pupil area"])

pipeline_xgb = Pipeline([("scaler", MinMaxScaler()), ("regressor", XGBRegressor())])
pipeline_xgb_params = pipeline_xgb.get_params().keys()

pipeline_rf = Pipeline(
    [("scaler", MinMaxScaler()), ("regressor", RandomForestRegressor())]
)
pipeline_rf_params = pipeline_rf.get_params().keys()

pipeline_ridge = Pipeline([("scaler", MinMaxScaler()), ("regressor", Ridge())])
pipeline_ridge_params = pipeline_ridge.get_params().keys()

hyperparameter_grid_xgb = {
    "regressor__n_estimators": [50, 100, 500, 1000, 2000],
    "regressor__max_depth": [3, 6, 9, 12, None],
    "regressor__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "regressor__colsample_bytree": [0.2, 0.4, 0.6, 0.8],
}

hyperparameter_grid_rf = {
    "regressor__n_estimators": [50, 100, 500, 1000, 2000],
    "regressor__max_depth": [3, 6, 9, 12, None],
}

hyperparameter_grid_ridge = {
    "regressor__alpha": [0, 1, 2, 5, 10, 20],
    "regressor__fit_intercept": [True, False],
}


random_cv_xgb = GridSearchCV(
    estimator=pipeline_xgb,
    param_grid=hyperparameter_grid_xgb,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=5,
    return_train_score=True,
)

random_cv_rf = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=hyperparameter_grid_rf,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=5,
    return_train_score=True,
)

random_cv_ridge = GridSearchCV(
    estimator=pipeline_ridge,
    param_grid=hyperparameter_grid_ridge,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=5,
    return_train_score=True,
)

random_cv_xgb.fit(X[:-150], y[:-150])
random_cv_rf.fit(X[:-150], y[:-150])
random_cv_ridge.fit(X[:-150], y[:-150])

print(
    f"Best xgboost: {random_cv_xgb.best_estimator_} | Score: {random_cv_xgb.best_score_}"
)
print(
    f"Best RandFor: {random_cv_rf.best_estimator_} | Score: {random_cv_rf.best_score_}"
)
print(
    f"Best ridgeRe: {random_cv_ridge.best_estimator_} | Score: {random_cv_ridge.best_score_}"
)

with open("../models/regression_models/initial_cv/xgb.pkl", "wb") as f:
    model = random_cv_xgb.best_estimator_
    model.fit(X, y)
    pkl.dump(model, f)

with open("../models/regression_models/initial_cv/rf.pkl", "wb") as f:
    model = random_cv_rf.best_estimator_
    model.fit(X, y)
    pkl.dump(model, f)

with open("../models/regression_models/initial_cv/ridge.pkl", "wb") as f:
    model = random_cv_ridge.best_estimator_
    model.fit(X, y)
    pkl.dump(model, f)
