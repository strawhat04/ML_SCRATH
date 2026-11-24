import numpy as np
from ml_scratch.regressor.boosting.GradientBoosting import GradientBoosting
from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    filter=lambda record: (
        record["name"].startswith("ml_scratch.")
        and record["level"].name == "INFO"
    ),
)

X_train = np.array([[1.], [2.], [3.], [4.]])
y_train = np.array([10., 20., 30., 40.])

X_val = np.array([[1.2], [2.8], [3.8]])
y_val = np.array([12., 28., 38.])

model = GradientBoosting(
    n_estimators=10,
    learning_rate=0.5,
    max_depth=1,
    min_sample_leaf=1,
    min_sample_split=2
)

model.fit(X_train, y_train, X_val, y_val)

print("Training predictions:", model.predict(X_train))
print("Validation predictions:", model.predict(X_val))

print("Training loss:", model.training_loss)
print("Validation loss:", model.validation_loss)

predictions = model.predict(X_val)

expected_mse = np.mean((y_val - predictions) ** 2)

recorded_mse = model.validation_loss[-1]

print("Expected MSE:", expected_mse)
print("Recorded MSE:", recorded_mse)

assert np.isclose(expected_mse, recorded_mse)