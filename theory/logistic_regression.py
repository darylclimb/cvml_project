import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons, make_classification

# Params
lr = 0.03
steps = 100


# Define the models
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 40),
                                    nn.LeakyReLU(),
                                    nn.Linear(40, 20),
                                    nn.LeakyReLU(),
                                    nn.Linear(20, 1),
                                    )

    def forward(self, x):
        return torch.sigmoid(self.layers(x))


if __name__ == '__main__':

    model_perceptron = Perceptron()
    model_mlp = MLP()

    # Datasets
    datasets = [
        make_classification(n_features=2, n_redundant=0, n_informative=2,
                            random_state=123, n_clusters_per_class=1),
        make_moons(shuffle=True, noise=0.05, random_state=123)
    ]

    # Loss
    criterion = torch.nn.BCELoss()

    fig = plt.figure(figsize=(12, 8))
    i = 1
    for data in datasets:
        X, y = data
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float().unsqueeze(-1)

        # Decision boundary visualization
        h = 0.01
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Xmesh = np.c_[xx.ravel(), yy.ravel()]

        for model in [model_perceptron, model_mlp]:
            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Train loop
            for step in range(steps):
                # Clear gradient
                optimizer.zero_grad()

                probs = model(X)
                loss = criterion(probs, y)

                # also get accuracy
                preds = (probs > 0.5).float()
                accuracy = (preds == y).sum() / y.shape[0]

                # Compute gradients
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    ax = plt.subplot(len(datasets), 2, i)
                    probs = model(torch.from_numpy(Xmesh).float())
                    Z = (probs > 0.5).float().reshape(xx.shape)
                    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
                    ax.scatter(X[:, 0], X[:, 1], c=y, s=30,
                               cmap=plt.cm.coolwarm, edgecolors='k', alpha=1)
                    ax.set_title(
                        f'{model.__class__.__name__} | Step: {step + 1} | Loss: {loss:.3f} | Acc: {accuracy:.2f}')
                    plt.tight_layout()

                    plt.pause(0.00001)
                    if step != steps - 1:
                        ax.cla()
            i += 1
    plt.show()
