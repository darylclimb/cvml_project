import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons

# Params
lr = 0.3
steps = 100

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)


# Define the model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc_out = nn.Linear(20, 1)

    def forward(self, x):
        """
        Forward propagation
            y1 = W_1 * X + b
            a1 = relu(y1)

            y2 = W_2 * a1 + b
            probs = sigmoid(y2)
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    # Loss
    criterion = torch.nn.BCELoss()

    # Datasets
    X, y = make_moons(n_samples=50, shuffle=True, noise=0.0, random_state=seed)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().unsqueeze(-1)

    # Experiments
    name = ['Weight Init = 0.1\n [Gradient vanish]',
            'Weight Init = 0\n [No learning]',
            'Kaiming Init',
            'Weight Init = 10\n [Gradient vanish due to Sigmoid]']

    init_funcs = [lambda x: nn.init.constant_(x, 0.1),
                  lambda x: nn.init.constant_(x, 0),
                  lambda x: x,  # By default kaiming uniform
                  lambda x: nn.init.constant_(x, 10)]

    # Decision boundary visualization
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    Xmesh = torch.from_numpy(Xmesh).float()

    fig = plt.figure(figsize=(14, 10))
    idx = 1
    for name, init_fn in zip(name, init_funcs):
        model = MLP()
        # Init weights and biased
        for layer in model.modules():
            if type(layer) == nn.Linear:
                init_fn(layer.weight)
                init_fn(layer.bias)

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=lr)

        grad_trajectories = []
        for p in model.parameters():
            grad_trajectories.append([[] for i in range(p.numel())])

        # Init figures
        ax1 = plt.subplot(len(init_funcs), 5, idx)
        ax2 = plt.subplot(len(init_funcs), 5, idx + 1)
        ax3 = plt.subplot(len(init_funcs), 5, idx + 2)
        ax4 = plt.subplot(len(init_funcs), 5, idx + 3)
        ax5 = plt.subplot(len(init_funcs), 5, idx + 4)

        # Train loop
        for step in range(steps):
            model.train()
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
                model.eval()
                # Plot weights and gradients
                for i, p in enumerate(model.parameters()):
                    grads = p.grad.detach().flatten()
                    params = p.detach().flatten()
                    for j, (per_layer_params, per_layer_grad) in enumerate(zip(params, grads)):
                        grad_trajectories[i][j].append(per_layer_grad.item())

                [ax1.plot(range(step + 1), x) for x in grad_trajectories[0]]
                ax1.set_title('Layer 1: dL/dw')
                ax1.grid()

                [ax2.plot(range(step + 1), x) for x in grad_trajectories[1]]
                ax2.set_title('Layer 1: dL/db')
                ax2.grid()

                [ax3.plot(range(step + 1), x) for x in grad_trajectories[2]]
                ax3.set_title('Layer 2: dL/dw')
                ax3.grid()

                [ax4.plot(range(step + 1), x) for x in grad_trajectories[3]]
                ax4.set_title('Layer 2: dL/db')
                ax4.grid()

                # Draw decision boundary
                probs_mesh = model(Xmesh)
                Z = (probs_mesh > 0.5).float().reshape(xx.shape)
                ax5.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
                ax5.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.coolwarm, edgecolors='k', alpha=1)
                ax5.set_title(name)
                ax5.set_xticks([])
                ax5.set_yticks([])
                ax5.set_xlabel(f'Step: {step + 1}')

                plt.tight_layout()
                plt.pause(0.00001)

                if step != steps - 1:
                    ax1.cla()
                    ax2.cla()
                    ax3.cla()
                    ax4.cla()
                    ax5.cla()
        idx += 5
    plt.show()
