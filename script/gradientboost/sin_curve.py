"""Compare gradient boosting configurations on a noisy sine curve."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.figure import Figure

from ml_scratch.regressor.boosting.GradientBoosting import GradientBoosting


BASELINE = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 1,
    "min_sample_leaf": 1,
    "min_sample_split": 2,
}

CONFIGURATIONS = {
    "Baseline": BASELINE,
    "Max depth 3": BASELINE | {"max_depth": 3},
    "Max depth 8": BASELINE | {"max_depth": 8},
    "Max depth 3 & rate 0.5": BASELINE | {"min_sample_leaf": 12, "learning_rate": 0.5},
}


def sine_signal(x: np.ndarray) -> np.ndarray:
    return 3 * np.sin(x) + 0.5 * x


def plot_comparison(configurations: dict[str, dict[str, int | float | None]]) -> tuple[Figure, Figure]:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, size=(500, 1))
    y = sine_signal(X[:, 0]) + rng.normal(0, 1.2, size=len(X))

    train_idx, val_idx, test_idx = np.split(rng.permutation(len(X)), [350, 425])
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_grid = np.linspace(0, 10, 500).reshape(-1, 1)

    loss_figure, loss_ax = plt.subplots(figsize=(12, 5), layout="constrained")
    fit_figure, fit_ax = plt.subplots(figsize=(12, 5), layout="constrained")
    fit_ax.scatter(X_test[:, 0], y_test, color="gray", alpha=0.45, label="Test observations")
    fit_ax.plot(X_grid[:, 0], sine_signal(X_grid[:, 0]), color="black", linestyle="--", label="True relationship")

    for name, parameters in configurations.items():
        model = GradientBoosting(**parameters)
        model.fit(X_train, y_train, X_val, y_val)

        rounds = np.arange(1, len(model.training_loss) + 1)
        style = {}
        if name == "Baseline":
            style = {"linewidth": 2.5, "zorder": 4, "marker": "o", "markersize": 4, "markevery": 0.1}

        (training_line,) = loss_ax.plot(rounds, model.training_loss / y_train.size, label=f"{name} (train)", **style)
        color = training_line.get_color()
        loss_ax.plot(rounds, model.validation_loss / y_val.size, color=color, linestyle="--", label=f"{name} (validation)", **style)
        fit_ax.plot(X_grid[:, 0], model.predict(X_grid), color=color, label=name, **style)

    loss_ax.set(title="Gradient boosting learning curves", xlabel="Boosting round", ylabel="Mean squared error")
    loss_ax.legend(ncols=2)
    loss_ax.grid(alpha=0.2)

    fit_ax.set(title="Predicted and true sine relationships", xlabel="x", ylabel="y")
    fit_ax.legend(ncols=2)
    fit_ax.grid(alpha=0.2)
    return loss_figure, fit_figure


if __name__ == "__main__":
    logger.disable("ml_scratch")
    try:
        plot_comparison(CONFIGURATIONS)
    finally:
        logger.enable("ml_scratch")
    plt.show()
