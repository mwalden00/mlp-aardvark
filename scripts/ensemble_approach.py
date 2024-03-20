import pickle as pkl
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV


def get_model(path):
    with open(path, "rb") as f:
        return pkl.load(f)


reg_ens_reg = XGBRegressor(n_estimators=100)

xgb = get_model("../models/regression_models/initial_cv/xgb.pkl")
ridge = get_model("../models/regression_models/initial_cv/ridge.pkl")
rf = get_model("../models/regression_models/initial_cv/rf.pkl")

lstm = get_model("../models/nn_models/lstm.pkl")
lstm_rf = get_model("../models/nn_models/rf_lstm.pkl")
lstm_xgb = get_model("../models/nn_models/xgb_lstm.pkl")
lstm_ridge = get_model("../models/nn_models/ridge_lstm.pkl")

with open("../data/processed/traj_and_pupil_data.pkl", "rb") as f:
    data = pkl.load(f)

X = torch.Tensor(np.concatenate(data["trajectories"], axis=1).T).reshape(100, 150, 13)[
    :, :100, :
]
y = torch.Tensor(np.concatenate(data["pupil area"])).reshape(100, 150)[:, :100]

X_train = X[:60]
X_val = X[60:]

y_train = y[:60]
y_val = y[60:]

# Get regression friendly shapes
X_train_reg = X_train.reshape((X_train.shape[0] * X_train.shape[1]), X_train.shape[2])
X_val_reg = X_val.reshape((X_val.shape[0] * X_val.shape[1]), X_val.shape[2])

y_train_reg = y_train.reshape(y_train.shape[0] * y_train.shape[1])
y_val_reg = y_val.reshape(y_val.shape[0] * y_val.shape[1])

# Fit + predict train
xgb.fit(X_train_reg, y_train_reg)
ridge.fit(X_train_reg, y_train_reg)
rf.fit(X_train_reg, y_train_reg)


def get_stacked_data_pred(X_train, X_train_reg, y_train_reg):
    y_train_reg_pred_xgb = torch.Tensor(xgb.predict(X_train_reg))
    y_train_reg_pred_rf = torch.Tensor(rf.predict(X_train_reg))
    y_train_reg_pred_ridge = torch.Tensor(ridge.predict(X_train_reg))

    # Predict with NN models
    y_train_pred_lstm = (
        lstm.forward(X_train).reshape(*y_train_reg.shape).detach().numpy()
    )

    y_train_pred_lstm_rf = (
        lstm_rf.forward(y_train_reg_pred_rf.reshape(*list(X_train.shape)[:2] + [1]))
        .reshape(*y_train_reg.shape)
        .detach()
        .numpy()
    )

    y_train_pred_lstm_xgb = (
        lstm_xgb.forward(y_train_reg_pred_xgb.reshape(*list(X_train.shape)[:2] + [1]))
        .reshape(*y_train_reg.shape)
        .detach()
        .numpy()
    )

    y_train_pred_lstm_ridge = (
        lstm_ridge.forward(
            y_train_reg_pred_ridge.reshape(*list(X_train.shape)[:2] + [1])
        )
        .reshape(*y_train_reg.shape)
        .detach()
        .numpy()
    )

    # Stack prediction + fit the ensemble
    stacked_y_train = np.stack(
        (
            y_train_pred_lstm,
            y_train_pred_lstm_rf,
            y_train_pred_lstm_xgb,
            y_train_pred_lstm_ridge,
            y_train_reg_pred_rf.detach().numpy(),
            y_train_reg_pred_xgb.detach().numpy(),
            y_train_reg_pred_ridge.detach().numpy(),
            np.arange(y_train_reg.shape[0]),
        ),
        axis=1,
    )

    return stacked_y_train


hyperparameter_grid = {
    "n_estimators": [50, 100, 500, 1000, 2000],
    "max_depth": [3, 6, 9, 12, None],
}

ens_cv = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=hyperparameter_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=5,
    return_train_score=True,
)

stacked_y_train = get_stacked_data_pred(X_train, X_train_reg, y_train_reg)

ens_cv.fit(stacked_y_train, y_train_reg)

ens_reg = ens_cv.best_estimator_
ens_reg.fit(stacked_y_train, y_train_reg)
reg_ens_reg.fit(stacked_y_train[:, 4:], y_train_reg)

stacked_y_val = get_stacked_data_pred(X_val, X_val_reg, y_val_reg)
y_val_pred = ens_reg.predict(stacked_y_val)
y_val_pred_reg = reg_ens_reg.predict(stacked_y_val[:, 4:])

with open("../models/ensemble.pkl", "wb") as f:
    pkl.dump(ens_reg, f)

print("Ensemble MSE:", mean_absolute_error(y_val_reg, y_val_pred))
print("Reg. Only Ensemble:", mean_absolute_error(y_val_reg, y_val_pred_reg))
