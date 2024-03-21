from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import src.mlp_nn as mlp_nn
from src.arg_extractor import get_args
import tqdm

with open("../data/processed/traj_and_pupil_data.pkl", "rb") as f:
    data = pkl.load(f)

if __name__ == "__main__":
    args = get_args()
    # Seed rng's
    np.random.RandomState(seed=args.seed)
    torch.manual_seed(seed=args.seed)

    num_epochs = args.num_epochs  # Num epochs

    gpu = args.use_gpu  # Gpu use
    cuda = args.cuda_num  # Cuda num
    device = "cuda" if gpu else "cpu"
    if cuda > -1 and gpu:
        device = f"cuda:{cuda}"

    with torch.device(device):
        # Get batched data (batched by stim exposure); remove the last 50 entries of each batch (to remove drift)

        X = torch.Tensor(data["trajectories"])[:, :100, :]
        y = torch.Tensor(np.stack(data["pupil area"]))[:, :100]

        X_train = X[:80]
        X_val = X[80:]

        y_train = y[:80]
        y_val = y[80:]

        loss = nn.L1Loss()

        reg_ens = Ridge()
        pipe = args.nn_type[:4] == "pipe"
        ens = args.nn_type[:3] == "ens" or pipe

        in_dim = 13

        if ens:
            affix = args.nn_type.split("_")[1]
            f = open("../models/regression_models/initial_cv/" + affix + ".pkl", "rb")
            reg = pkl.load(f)
            f.close()
            train_shape = (X_train.shape[0] * X_train.shape[1], 13)
            reg.fit(
                X_train.reshape(*train_shape).cpu(),
                y_train.reshape(train_shape[0]).cpu(),
            )
            val_shape = (X_val.shape[0] * X_val.shape[1], 13)

            if pipe:
                X_train = torch.Tensor(
                    reg.predict(X_train.reshape(*train_shape).cpu()).reshape(
                        list(X_train.shape)[:2] + [1]
                    )
                ).to(device=device)
                X_val = torch.Tensor(
                    reg.predict(X_val.reshape(*val_shape).cpu()).reshape(
                        list(X_val.shape)[:2] + [1]
                    )
                ).to(device=device)
                in_dim = 1
            else:
                lin = nn.Linear(2, 1)

        num_layers = args.layers
        hidden_dim = args.hidden_dim
        bias = args.bias
        dropout = args.dropout
        model = mlp_nn.base_LSTM(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_layers,
            out_dim=1,
            bias=bias,
            dropout=dropout,
        ).to(device)

        # Adam Params
        weight_decay = args.weight_decay_coefficient
        lr = args.learning_rate
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False
        )
        lr_schedular = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=2e-5
        )

        train_losses = []
        val_losses = []

        best_val_loss = np.inf

        with tqdm.tqdm(range(num_epochs)) as pbar:
            for epoch in pbar:
                train_loss = []
                for n in torch.randperm(40) * 2:
                    batch = X_train[n - 1 : n]
                    # Train data + Optim
                    # print(X_train.shape)
                    out = model.forward(batch).reshape(*y_train[n - 1 : n].shape)

                    optimizer.zero_grad()

                    l_train = loss(out, y_train[n - 1 : n])
                    l_train.backward()

                    optimizer.step()
                    lr_schedular.step()

                    train_loss.append(l_train.cpu().item())
                train_losses.append(train_loss)

                # Validation data
                out = model.forward(X_val).reshape(*y_val.shape)

                l_val = loss(out, y_val)

                val_losses.append(l_val.cpu().item())
                pbar.set_description(
                    "Train loss: {:.4f} | Val loss: {:.4f}".format(l_train, l_val)
                )

        np.array(train_losses).tofile(args.exp + "_train_loss.csv", sep=",")
        np.array(val_losses).tofile(args.exp + "_val_loss.csv", sep=",")

        with open(f"../models/nn_models/{args.exp}.pkl", "wb") as f:
            pkl.dump(model, f)
