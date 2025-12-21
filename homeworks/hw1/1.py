# for case if the problem with cvxpy wouldn't sovled
import math
import time
import warnings
from functools import partial
from itertools import product
from typing import Callable, Iterator, Optional, List, Tuple, Union

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Plotting functions
def moving_average(x, w):
    return scipy.signal.savgol_filter(x, w, min(3, w - 1))


def draw_plots(
    data: dict,
    plots: List[Tuple[dict, dict]],
    title: str = "",
    row_plots: int = 2,
    plot_width: float = 8,
    plot_height: float = 4,
    use_common_legend: bool = True,
):
    num_plots = len(plots)
    row_plots = min(row_plots, num_plots)
    column_plots = math.ceil(num_plots / row_plots)

    fig, axs = plt.subplots(
        column_plots,
        row_plots,
        figsize=(plot_width * row_plots, plot_height * column_plots),
    )
    if len(title):
        fig.suptitle(title, fontsize=14)
    axs_list = [axs] if num_plots == 1 else list(axs.flat)
    for ax in axs_list:
        ax.grid()
        ax.set_visible(False)

    for ax, (p1, p2) in zip(axs_list, plots):
        ax.set_visible(True)

        x_label = p1.get("axis_name", p1["name"])
        y_label = p2.get("axis_name", p2["name"])
        if p1.get("log", False):
            ax.set_xscale("log")
            x_label = f"{x_label}, log scale"
        if p2.get("log", False):
            ax.set_yscale("log")
            y_label = f"{y_label}, log scale"

        ax.set_title(f"{p2['name']} over {p1['name']}")
        ax.set(xlabel=x_label, ylabel=y_label)

        scatter_all = p1.get("scatter", False) or p2.get("scatter", False)

        for method, method_data in data.items():
            label = method

            x_values = method_data[p1.get("ref", p1["name"])]
            y_values = method_data[p2.get("ref", p2["name"])]

            y_smooth_w = p2.get("smooth", 0)
            if y_smooth_w:
                y_smooth = moving_average(y_values, w=y_smooth_w)
                ax.fill_between(
                    x_values,
                    scipy.ndimage.minimum_filter1d(y_smooth, y_smooth_w),
                    scipy.ndimage.maximum_filter1d(y_smooth, y_smooth_w),
                    alpha=0.1,
                )
                y_values = y_smooth

            ax.plot(x_values, y_values, label=label)
            ax_color = ax.get_lines()[-1].get_color()
            if scatter_all:
                ax.scatter(x_values, y_values, s=15, color=ax_color)
            else:
                ax.scatter(x_values[-1], y_values[-1], s=15, color=ax_color)

    if use_common_legend:
        lines_labels = [axs_list[0].get_legend_handles_labels()]
        lines, labels = [sum(x, []) for x in zip(*lines_labels)]
        fig.legend(
            lines,
            labels,
            scatterpoints=1,
            markerscale=3,
            loc="outside lower center",
            ncol=min(6, len(data)),
            bbox_to_anchor=(0.5, -0.05 * (math.ceil(len(data) / 6) + 1)),
        )
    else:
        if len(data) > 1:
            for ax in axs_list:
                ax.legend()

    plt.tight_layout()
    plt.show()

# Mushrooms dataset setup
import os
import urllib.request

dataset_url = "https://drive.google.com/uc?id=1lgwawQxGD_6XruWbquMH6W2yKPVQ5kdi"
dataset_path = "mushrooms.txt"

if not os.path.exists(dataset_path):
    print("Downloading mushrooms dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print("Download complete!")
else:
    print("Mushrooms dataset already exists.")

# Load the dataset
dataset = dataset_path
data = load_svmlight_file(dataset)
mushrooms_x, mushrooms_y = data[0].toarray(), data[1]

# Make y -1 or 1
mushrooms_y = 2 * mushrooms_y - 3

train_mush_x, test_mush_x, train_mush_y, test_mush_y = train_test_split(
    mushrooms_x, mushrooms_y, test_size=0.2, random_state=42
)

print(f"{train_mush_x.shape=}")
print(f"{train_mush_y.shape=}")

train_mush = list(zip(train_mush_x, train_mush_y))
test_mush = list(zip(test_mush_x, test_mush_y))

print(f"{len(train_mush)=}")
print(f"{len(test_mush)=}")

assert len(train_mush) == 6499
assert len(test_mush) == 1625

# Calculate constants for the optimization problem
n = train_mush_x.shape[0]
L_unregularized = np.sum(np.linalg.norm(train_mush_x, axis=1)**2) / (4 * n)
lambda_ = L_unregularized / 1000
L = L_unregularized + lambda_

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function to compute the objective value
def mush_f(w, x=train_mush_x, y=train_mush_y, lambda_=lambda_):
    n = x.shape[0]
    z = y * (x @ w)
    log_likelihood = np.log(1 + np.exp(-z))
    return np.mean(log_likelihood) + 0.5 * lambda_ * np.dot(w, w)

# Function to compute the gradient
def mush_grad(w, x=train_mush_x, y=train_mush_y, lambda_=lambda_):
    n = x.shape[0]
    z = y * (x @ w)
    grad = -(y * sigmoid(-z)) @ x / n
    return grad + lambda_ * w

# Function to compute the Hessian
def mush_hess(w, x=train_mush_x, y=train_mush_y, lambda_=lambda_):
    n = x.shape[0]
    z = y * (x @ w)
    s = sigmoid(-z) * sigmoid(z)
    S = np.diag(s)
    hessian = (x.T @ S @ x) / n
    return hessian + lambda_ * np.eye(x.shape[1])

# Function to compute accuracy
def mush_accuracy(w, x=test_mush_x, y=test_mush_y):
    preds = sigmoid(x @ w)
    preds = np.where(preds >= 0.5, 1, -1)
    accuracy = np.mean(preds == y)
    return accuracy

# Assertion function for testing
def assert_mush(mush_f: Callable, mush_grad: Callable, mush_hess: Callable):
    w = np.zeros(train_mush_x[0].shape[0])
    assert np.isclose(mush_f(w), 0.6931471805599453)
    assert np.isclose(mush_grad(w).sum(), -0.3732112632712724)
    assert np.isclose(mush_hess(w).sum(), 110.83858858858859)

# Test the functions
assert_mush(mush_f, mush_grad, mush_hess)

# Base classes for optimization methods
class BaseSolver:
    def step(self, x: np.ndarray, k: int) -> np.ndarray:
        # This function should be overridden by inherited classes
        raise NotImplementedError


class ManualSolver(BaseSolver):
    def __init__(
        self,
        lr: Union[float, Callable],
        grad_f: Callable = mush_grad,
        hess_f: Callable = mush_hess,
    ) -> None:
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.lr = lr if isinstance(lr, Callable) else lambda _: lr

# Function to test optimization methods
def check_approx_mush(
    approximations: List[Tuple[str, BaseSolver]],
    epochs: int = 500,
    start_w: np.ndarray = None,
    stop_criterion: float = 1e-5,
) -> dict:
    if start_w is None:
        w_mush_shape = train_mush_x.shape[1]
        np.random.seed(420)
        start_w = np.random.randn(w_mush_shape)

    results_dict = {}

    for name, approx in approximations:
        accuracies = []
        criterion_logs = []  # gradient norms or other criteria
        time_logs = []

        np.random.seed(420)
        w = start_w.copy()

        start_time = time.time()

        with tqdm(range(epochs), desc=name) as loop:
            for k in loop:
                # Evaluate before training (for k=0, this is the initial evaluation)
                loss = mush_f(w)
                accuracy = mush_accuracy(w)
                grad_norm = np.linalg.norm(mush_grad(w))

                # Log values
                accuracies.append(accuracy)
                criterion_logs.append(grad_norm)
                time_logs.append(time.time() - start_time)
                
                loop.set_postfix({"Loss": loss, "Accuracy": accuracy, "Grad norm": grad_norm})

                # Stop if criterion <= stop_criterion
                if grad_norm <= stop_criterion:
                    break
                
                # Train
                w = approx.step(w, k)

                # Stop if the function value is undefined
                if np.isnan(loss) or np.isinf(loss):
                    break

        results_dict[name] = {
            "Epoch": list(range(len(accuracies))),
            "Time": time_logs,
            "Accuracy": accuracies,
            "Criterion": criterion_logs,
            "W": w.copy(),
        }

    return results_dict

# Gradient Descent implementation
class GradientDescent(ManualSolver):
    def __init__(self, lr: float, grad_f: Callable = mush_grad, hess_f: Callable = mush_hess) -> None:
        super().__init__(lr=lr, grad_f=grad_f, hess_f=hess_f)
        self.lr_value = lr  # Store the actual learning rate value

    def step(self, w: np.ndarray, k: int) -> np.ndarray:
        grad = self.grad_f(w)
        return w - self.lr_value * grad

# Test Gradient Descent
w_mush_shape = train_mush_x.shape[1]
np.random.seed(420)
start_w_mush = np.random.randn(w_mush_shape)

gd_solver = GradientDescent(lr=0.001)

results = check_approx_mush([("Gradient Descent", gd_solver)], epochs=500, start_w=start_w_mush, stop_criterion=1e-5)

plot_criterion_epoch = partial(
    draw_plots,
    plots=[
        ({"name": "Epoch"}, {"name": "Criterion"}),
        ({"name": "Epoch"}, {"name": "Criterion", "log": True})
    ],
)

plot_criterion_epoch(results)

# Test with different learning rates
# Test with 1/L
gamma_L = 1.0 / L
gd_solver_L = GradientDescent(lr=gamma_L)
results_L = check_approx_mush([("Gradient Descent 1/L", gd_solver_L)], epochs=500, start_w=start_w_mush, stop_criterion=1e-5)
print(f"gamma_L = {gamma_L}")

# Test with different constant steps
gammas = [0.1/L, 0.5/L, 1.0/L, 2.0/L, 3.0/L]
solvers = [(f"GD gamma=${gamma:.1f} / L$", GradientDescent(lr=gamma)) for gamma in gammas]

results_constant = check_approx_mush(solvers, epochs=1000, start_w=start_w_mush, stop_criterion=1e-5)

# Plot results for constant step sizes
plot_constant_lr = partial(
    draw_plots,
    plots=[
        ({"name": "Epoch"}, {"name": "Criterion", "log": True}),
        ({"name": "Time"}, {"name": "Criterion", "log": True}),
        ({"name": "Epoch"}, {"name": "Accuracy"}),
        ({"name": "Time"}, {"name": "Accuracy"}),
    ],
)

plot_constant_lr(results_constant)

# Find the best step size based on final accuracy and gradient norm
best_method = None
best_accuracy = 0
best_grad_norm = float('inf')

for method, data in results_constant.items():
    final_accuracy = data["Accuracy"][-1]
    final_grad_norm = data["Criterion"][-1]

    if final_accuracy > best_accuracy or (final_accuracy == best_accuracy and final_grad_norm < best_grad_norm):
        best_accuracy = final_accuracy
        best_grad_norm = final_grad_norm
        best_method = method

print(f"Best method: {best_method} with accuracy: {best_accuracy:.4f} and grad norm: {best_grad_norm:.6f}")

# Gradient Descent with decay
class GradientDescentDecay(ManualSolver):
    def __init__(self, gamma: float, delta: float, grad_f: Callable = mush_grad, hess_f: Callable = mush_hess) -> None:
        super().__init__(lr=lambda k: gamma / (k + delta), grad_f=grad_f, hess_f=hess_f)
        self.gamma = gamma
        self.delta = delta

    def step(self, w: np.ndarray, k: int) -> np.ndarray:
        grad = self.grad_f(w)
        lr = self.lr(k)
        return w - lr * grad

# Test with different decay parameters
gamma_values = [5.25525, 10.5105]  # Some example values
delta_values = [0.5, 1.0]

decay_solvers = []
for gamma in gamma_values:
    for delta in delta_values:
        decay_solvers.append((f"GD decay gamma={gamma}, delta={delta}",
                             GradientDescentDecay(gamma=gamma, delta=delta)))

results_decay = check_approx_mush(decay_solvers, epochs=1000, start_w=start_w_mush, stop_criterion=1e-5)

# Gradient Descent with sqrt decay
class GradientDescentSqrtDecay(ManualSolver):
    def __init__(self, gamma: float, delta: float, grad_f: Callable = mush_grad, hess_f: Callable = mush_hess) -> None:
        super().__init__(lr=lambda k: gamma / (np.sqrt(k + delta)), grad_f=grad_f, hess_f=hess_f)
        self.gamma = gamma
        self.delta = delta

    def step(self, w: np.ndarray, k: int) -> np.ndarray:
        grad = self.grad_f(w)
        lr = self.lr(k)
        return w - lr * grad

# Test with different sqrt decay parameters
sqrt_decay_solvers = []
for gamma in gamma_values:
    for delta in delta_values:
        sqrt_decay_solvers.append((f"GD sqrt decay gamma={gamma}, delta={delta}",
                                  GradientDescentSqrtDecay(gamma=gamma, delta=delta)))

results_sqrt_decay = check_approx_mush(sqrt_decay_solvers, epochs=1000, start_w=start_w_mush, stop_criterion=1e-5)

# Compare all methods
all_methods = list(results_constant.items()) + list(results_decay.items()) + list(results_sqrt_decay.items())
all_results = dict(all_methods)

plot_all_methods = partial(
    draw_plots,
    plots=[
        ({"name": "Epoch"}, {"name": "Criterion", "log": True}),
        ({"name": "Time"}, {"name": "Criterion", "log": True}),
        ({"name": "Epoch"}, {"name": "Accuracy"}),
        ({"name": "Time"}, {"name": "Accuracy"}),
    ],
)

plot_all_methods(all_results)

# Heavy Ball method
class HeavyBall(ManualSolver):
    def __init__(self, lr: float, beta: float, grad_f: Callable = mush_grad, hess_f: Callable = mush_hess) -> None:
        super().__init__(lr=lr, grad_f=grad_f, hess_f=hess_f)
        self.lr_value = lr  # Store the actual learning rate value
        self.beta = beta
        self.prev_w = None

    def step(self, w: np.ndarray, k: int) -> np.ndarray:
        grad = self.grad_f(w)
        
        if k == 0:
            # First step, just use gradient descent
            self.prev_w = w.copy()
            return w - self.lr_value * grad
        else:
            # Subsequent steps, use momentum
            new_w = w - self.lr_value * grad + self.beta * (w - self.prev_w)
            self.prev_w = w.copy()
            return new_w

# Test Heavy Ball with different parameters
gamma_values = [0.19028590457161884, 0.3805718091432377, 0.5708577137148565]  # Different multiples of 1/L
beta_values = [0.5, 0.9]

hb_solvers = []
for gamma in gamma_values:
    for beta in beta_values:
        hb_solvers.append((f"Heavy Ball gamma={gamma}, beta={beta}",
                          HeavyBall(lr=gamma, beta=beta)))

results_hb = check_approx_mush(hb_solvers, epochs=500, start_w=start_w_mush, stop_criterion=1e-5)

# Nesterov Momentum method
class Nesterov(ManualSolver):
    def __init__(self, lr: float, beta: float, grad_f: Callable = mush_grad, hess_f: Callable = mush_hess) -> None:
        super().__init__(lr=lr, grad_f=grad_f, hess_f=hess_f)
        self.lr_value = lr  # Store the actual learning rate value
        self.beta = beta
        self.prev_w = None

    def step(self, w: np.ndarray, k: int) -> np.ndarray:
        if k == 0:
            # First step, just use gradient descent
            self.prev_w = w.copy()
            grad = self.grad_f(w)
            return w - self.lr_value * grad
        else:
            # Lookahead point
            lookahead = w + self.beta * (w - self.prev_w)
            grad = self.grad_f(lookahead)
            new_w = lookahead - self.lr_value * grad
            self.prev_w = w.copy()
            return new_w

# Test Nesterov with different parameters
nesterov_solvers = []
for gamma in gamma_values:
    for beta in beta_values:
        nesterov_solvers.append((f"Nesterov gamma={gamma}, beta={beta}",
                                Nesterov(lr=gamma, beta=beta)))

results_nesterov = check_approx_mush(nesterov_solvers, epochs=500, start_w=start_w_mush, stop_criterion=1e-5)

# Newton's method
class Newton(ManualSolver):
    def __init__(self, lr: float, grad_f: Callable = mush_grad, hess_f: Callable = mush_hess) -> None:
        super().__init__(lr=lr, grad_f=grad_f, hess_f=hess_f)
        self.lr_value = lr  # Store the actual learning rate value

    def step(self, w: np.ndarray, k: int) -> np.ndarray:
        grad = self.grad_f(w)
        hess = self.hess_f(w)
        
        try:
            # Solve Hessian * direction = gradient
            direction = np.linalg.solve(hess, grad)
            return w - self.lr_value * direction
        except np.linalg.LinAlgError:
            # If Hessian is singular, fall back to gradient descent
            return w - self.lr_value * grad

# Test Newton's method with different step sizes
newton_gammas = [1.0, 0.1/L, 0.5/L, 1.0/L, 2.0/L]

newton_solvers = []
for gamma in newton_gammas:
    if gamma == 1.0:
        newton_solvers.append((f"Newton gamma=1", Newton(lr=gamma)))
    else:
        newton_solvers.append((f"Newton gamma=${gamma:.1f} / L$", Newton(lr=gamma)))

results_newton = check_approx_mush(newton_solvers, epochs=300, start_w=start_w_mush, stop_criterion=1e-5)

# Test Newton's method with different starting points
start_points = [
    ("Zero vector", np.zeros(w_mush_shape)),
    ("Ones vector", np.ones(w_mush_shape)),
    ("Random vector", start_w_mush)
]

newton_start_results = {}
for name, start_point in start_points:
    result = check_approx_mush([("Newton", Newton(lr=1.0/L))], epochs=100, start_w=start_point, stop_criterion=1e-5)
    newton_start_results[f"Newton from {name}"] = result["Newton"]

plot_newton_start = partial(
    draw_plots,
    plots=[
        ({"name": "Epoch"}, {"name": "Criterion", "log": True}),
        ({"name": "Time"}, {"name": "Criterion", "log": True}),
        ({"name": "Epoch"}, {"name": "Accuracy"}),
        ({"name": "Time"}, {"name": "Accuracy"}),
    ],
)

plot_newton_start(newton_start_results)

# BFGS method
class BFGS(ManualSolver):
    def __init__(self, lr: float, grad_f: Callable = mush_grad, hess_f: Callable = mush_hess) -> None:
        super().__init__(lr=lr, grad_f=grad_f, hess_f=hess_f)
        self.lr_value = lr  # Store the actual learning rate value
        self.H = None  # Approximate inverse Hessian

    def step(self, w: np.ndarray, k: int) -> np.ndarray:
        grad = self.grad_f(w)
        
        if k == 0:
            # Initialize H as identity matrix
            self.H = np.eye(len(w))
            self.prev_w = w.copy()
            self.prev_grad = grad.copy()
            return w - self.lr_value * self.H @ grad
        else:
            # BFGS update
            s = w - self.prev_w
            y = grad - self.prev_grad
            
            # Avoid division by zero
            rho = 1.0 / (y @ s)
            if np.isinf(rho) or np.isnan(rho):
                rho = 0.0
            
            # BFGS update formula
            I = np.eye(len(w))
            self.H = (I - rho * np.outer(s, y)) @ self.H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            
            self.prev_w = w.copy()
            self.prev_grad = grad.copy()
            
            return w - self.lr_value * self.H @ grad

# Test BFGS with different step sizes
bfgs_gammas = [1.0/L, 5.0/L, 10.0/L, 20.0/L, 30.0/L]

bfgs_solvers = []
for gamma in bfgs_gammas:
    bfgs_solvers.append((f"BFGS gamma=${gamma:.0f} / L$", BFGS(lr=gamma)))

results_bfgs = check_approx_mush(bfgs_solvers, epochs=1000, start_w=start_w_mush, stop_criterion=1e-5)

# Find the best method from each category
best_gd = None
best_gd_accuracy = 0
for method, data in results_constant.items():
    final_accuracy = data["Accuracy"][-1]
    if final_accuracy > best_gd_accuracy:
        best_gd_accuracy = final_accuracy
        best_gd = (method, data)

best_hb = None
best_hb_accuracy = 0
for method, data in results_hb.items():
    final_accuracy = data["Accuracy"][-1]
    if final_accuracy > best_hb_accuracy:
        best_hb_accuracy = final_accuracy
        best_hb = (method, data)

best_nesterov = None
best_nesterov_accuracy = 0
for method, data in results_nesterov.items():
    final_accuracy = data["Accuracy"][-1]
    if final_accuracy > best_nesterov_accuracy:
        best_nesterov_accuracy = final_accuracy
        best_nesterov = (method, data)

best_newton = None
best_newton_accuracy = 0
for method, data in results_newton.items():
    final_accuracy = data["Accuracy"][-1]
    if final_accuracy > best_newton_accuracy:
        best_newton_accuracy = final_accuracy
        best_newton = (method, data)

# Compare the best methods
best_methods = {
    best_gd[0]: best_gd[1],
    best_hb[0]: best_hb[1],
    best_nesterov[0]: best_nesterov[1],
    best_newton[0]: best_newton[1]
}

plot_best_methods = partial(
    draw_plots,
    plots=[
        ({"name": "Epoch"}, {"name": "Criterion", "log": True}),
        ({"name": "Time"}, {"name": "Criterion", "log": True}),
        ({"name": "Epoch"}, {"name": "Accuracy"}),
        ({"name": "Time"}, {"name": "Accuracy"}),
    ],
)

plot_best_methods(best_methods)

print("Final comparison of best methods:")
for method, data in best_methods.items():
    final_accuracy = data["Accuracy"][-1]
    final_grad_norm = data["Criterion"][-1]
    epochs = len(data["Epoch"]) - 1
    time_taken = data["Time"][-1] if data["Time"] else 0
    print(f"{method}: Accuracy={final_accuracy:.4f}, Grad Norm={final_grad_norm:.6f}, Epochs={epochs}, Time={time_taken:.2f}s")
