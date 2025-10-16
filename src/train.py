import os

os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde

dde.config.set_default_float("float64")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    print("CUDA available")
    torch.set_default_device("cuda")
    device = torch.device("cuda")
else:
    print("CUDA *not* available")
    device = torch.device("cpu")
os.environ["DDE_BACKEND"] = "pytorch"
backend = os.environ["DDE_BACKEND"]
print(backend)


def net(input, output, hidden, num_hidden):
    net = dde.nn.FNN(
        [input] + [hidden] * num_hidden + [output],
        "tanh",
        "Glorot uniform",
        #  regularization="l2"
    )
    return net


def exact_solution(d, w0, t):
    """Defines the analytical solution to the under-damped harmonic oscillator
    problem above."""
    assert d < w0
    w = torch.sqrt(w0**2 - d**2)
    phi = torch.arctan(-d / w)
    A = 1 / (2.0 * torch.cos(phi))
    cos = torch.cos(phi + w * t)
    exp = torch.exp(-d * t)
    u = exp * 2.0 * A * cos
    return u


# Define the residual
def residual(input, output):
    dxdt = dde.grad.jacobian(output, input, i=0)
    dx2dt2 = dde.grad.jacobian(dxdt, input, i=0)

    return torch.as_tensor(m * dx2dt2 + mu * dxdt + k * output)


def custom_loss(y_true, y_pred):
    # print(y_true.shape)
    # print(y_pred.shape)
    loss = torch.mean((y_true - y_pred) ** 2)
    # print(loss.shape)
    return loss


if __name__ == "__main__":
    # Constants
    d, w0 = torch.as_tensor([2.0], device=device), torch.as_tensor(
        [20.0], device=device
    )
    mu, k = 2.0 * d, w0**2.0
    m = torch.as_tensor([1.0], device=device)
    # print(d.device)

    # Define the domain and coundary conditions
    time_domain = dde.geometry.TimeDomain(0, 2)
    bc_l = dde.icbc.NeumannBC(
        time_domain,
        lambda x: 0,
        lambda t, on_boundary: on_boundary and dde.utils.isclose(t[0], 0),
    )
    bc_r = dde.icbc.IC(time_domain, lambda x: 1, lambda _, on_initial: on_initial)

    # Create the neural network
    pinn = net(1, 1, hidden=75, num_hidden=4)

    # Compile data and model
    data = dde.data.TimePDE(
        time_domain,
        residual,
        [bc_l, bc_r],
        num_domain=5000,
        num_boundary=10,
        num_test=100,
    )
    model = dde.Model(data, pinn)
    # model.outputs_modify(lambda x, y: x * y)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    model.compile(
        optimizer=optimizer,
        loss_weights=[1e-4, 1e-1, 1],
        loss=lambda y_true, y_pred: custom_loss(y_true, y_pred),
    )

    # Train model
    losshistory, train_state = model.train(iterations=6000)
    # model.compile("L-BFGS")
    # losshistory, train_state = model.train()

    # Visualization
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    t_test = torch.linspace(0, 5, 300, device=device).view(-1, 1)
    u_exact = exact_solution(d, w0, t_test)
    t_test = t_test.cpu().numpy()
    u_exact = u_exact.cpu().numpy()
    u_pred = model.predict(t_test)
    plt.plot(t_test, u_exact)
    plt.plot(t_test, u_pred)
    plt.show()
