import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification

# Params
lr = 0.03
steps = 150
lambda_term = 0.01  # reg weight
dropout_rate = 0.5


# Define the model
class MLP(nn.Module):
    def __init__(self, dropout: float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.fc1 = nn.Linear(2, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc_out = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x) if self.dropout else x
        x = F.relu(x)

        x = self.fc2(x)
        x = self.dropout(x) if self.dropout else x
        x = F.relu(x)

        x = self.fc_out(x)
        return torch.sigmoid(x)


def bce(x, y, lambda_term=None, model=None):
    return F.binary_cross_entropy(x, y)


def bce_l1(x, y, lambda_term, model=None):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return F.binary_cross_entropy(x, y) + lambda_term * l1_norm


def bce_l2(x, y, lambda_term, model=None):
    l2_norm = sum(p.abs().sum() for p in model.parameters())
    return F.binary_cross_entropy(x, y) + lambda_term * l2_norm


if __name__ == '__main__':
    # Experiments
    names = ['Overfit', 'L1reg', 'L2reg', 'Dropout']
    losses = [bce, bce_l1, bce_l2, bce]
    models = [MLP(), MLP(), MLP(), MLP(dropout_rate)]

    # Datasets
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=511, n_clusters_per_class=2)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().unsqueeze(-1)

    # Decision boundary visualization
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    # Optimizer

    fig = plt.figure(figsize=(12, 10))
    i = 1
    for name, criterion, model in zip(names, losses, models):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train loop
        for step in range(steps):
            model.train()
            # Clear gradient
            optimizer.zero_grad()

            probs = model(X)

            loss = criterion(probs, y, lambda_term, model)

            # also get accuracy
            preds = (probs > 0.5).float()
            accuracy = (preds == y).sum() / y.shape[0]

            # Compute gradients
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                ax = plt.subplot(4, 2, i)
                probs = model(torch.from_numpy(Xmesh).float())
                Z = (probs > 0.5).float().reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
                ax.scatter(X[:, 0], X[:, 1], c=y, s=30,
                           cmap=plt.cm.coolwarm, edgecolors='k', alpha=1)
                ax.set_title(
                    f'{name} | Step: {step + 1} | Loss: {loss:.3f} | Acc: {accuracy:.2f}')
                ax.set_xticks([])
                ax.set_yticks([])

                params = []
                for p in model.parameters():
                    params.extend(p.detach().flatten().data)

                ax1 = plt.subplot(4, 2, i + 1)
                ax1.hist(x=np.array(params), bins=100)
                ax1.set_title(f'Weight Hist: Number of weights = {len(params)}')
                ax1.set_yscale('log')
                plt.tight_layout()
                plt.pause(0.00001)

                if step != steps - 1:
                    ax.cla()
                    ax1.cla()
        i += 2

    plt.show()
