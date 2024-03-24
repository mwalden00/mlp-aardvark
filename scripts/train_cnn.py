import pickle as pkl
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import one_layer_cnn as cnn
from arg_extractor import get_args
import tqdm


class ElecDataset(Dataset):
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        item = self.feature[idx]
        label = self.target[idx]

        return item, label


def train_model(model, train_loader):

    running_loss = 0.0

    model.train()

    for _, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.reshape(inputs.shape[2], inputs.shape[0] * inputs.shape[1]).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss

    train_loss = running_loss / len(train_loader)

    return train_loss


def valid_model(model, valid_loader):
    running_loss = 0.0

    model.eval()

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.reshape(inputs.shape[2], inputs.shape[0] * inputs.shape[1]).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds, labels)
            running_loss += loss

        valid_loss = running_loss / len(valid_loader)
    return valid_loss


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

        X = torch.Tensor(data["trajectories"])[:, :100, :]
        y = torch.Tensor(np.stack(data["pupil area"]))[:, :100]

        X_train = X[:80]
        X_val = X[80:]

        y_train = y[:80]
        y_val = y[80:]
        
        train = ElecDataset(X_train.reshape(X_train.shape[0] , X_train.shape[1], 13), y_train)
        valid = ElecDataset(X_val.reshape(X_val.shape[0] , X_val.shape[1], 13), y_val)
        train_loader = DataLoader(train, batch_size=1, shuffle=False)
        valid_loader = DataLoader(valid, batch_size=1, shuffle=False)
        
        # loss = nn.L1Loss()
        # in_dim = 13
        # bias = args.bias
        # dropout = args.dropout

        lr = args.learning_rate
        model = cnn.CNN_Net((13, 100), 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        lr_schedular = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=2e-5
        )

        train_losses = []
        valid_losses = []

        best_val_loss = np.inf

        with tqdm.tqdm(range(num_epochs)) as pbar:
            for epoch in pbar:
                train_loss = train_model(model, train_loader)
                train_losses.append(train_loss.detach().numpy())
                valid_loss = valid_model(model, valid_loader)
                valid_losses.append(valid_loss.detach().numpy())

        np.array(train_losses).tofile(args.exp + "_train_loss.csv", sep=",")
        np.array(valid_losses).tofile(args.exp + "_val_loss.csv", sep=",")
        with open(f"../models/nn_models/{args.exp}.pkl", "wb") as f:
            pkl.dump(model, f)
