"""
A simple script to view various activations and their gradient
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def swish(input):
    return x * torch.sigmoid(input)


def mish(input):
    return input * torch.tanh(F.softplus(input))


def forward(x: torch.Tensor, activation_fn):
    x.requires_grad_()
    act = activation_fn(x)
    act.backward(torch.ones_like(x))
    grad = x.grad.clone()
    # Zero grad
    x.grad.zero_()
    x.detach_()
    return grad, act.detach()


if __name__ == '__main__':
    activation_functions = [
        torch.sigmoid,
        torch.tanh,
        F.relu,
        F.leaky_relu,
        F.gelu,
        mish,
        swish
    ]

    for act_fn in activation_functions:
        x = torch.arange(-5, 5, 0.1)

        grad, y = forward(x, act_fn)
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, grad)
        plt.title(act_fn.__name__)
        plt.legend([f'z = {act_fn.__name__}(x)', 'gradient'])

    plt.show()
