from dataclasses import dataclass, field
from loguru import logger
import numpy as np

from ml_scratch.regressor.tree.DecisionTree import DecisionTreeRegressor

@dataclass
class IndividualTree:
    model : DecisionTreeRegressor

@dataclass
class GradientBoosting:
    n_estimators: int = 1
    learning_rate : float = 0.5
    max_depth: int = 3
    min_sample_leaf: int = 1
    min_sample_split: int = 2
    max_features: int | None = None

    def __post_init__(self):
        self.random_trees : list[IndividualTree] = []
        self.training_loss : np.ndarray = np.full(self.n_estimators, np.inf)
        self.validation_loss : np.ndarray = np.full(self.n_estimators, np.inf)

    def fit(self, X:np.ndarray | list, Y: np.ndarray | list, X_val : np.ndarray | list | None = None, Y_val : np.ndarray | list | None = None) -> None:

        Y_train = np.array(Y)
        X_train = np.array(X)

        self.init_pred = self.initial_prediction(Y_train)

        y0_train = np.full(Y_train.shape[0], self.init_pred)
        ym_train = np.zeros(Y_train.shape[0])

        self.random_trees =[]
        self.training_loss = np.full(self.n_estimators, np.inf)
        self.validation_loss = np.full(self.n_estimators, np.inf)

        if X_val is not None or Y_val is not None: 
            X_val = np.array(X_val)
            Y_val = np.array(Y_val)
            y0_val = np.full(Y_val.shape[0], self.init_pred)
            ym_val = np.zeros(Y_val.shape[0])

        for idx in range(self.n_estimators):

            ym_train_1 = y0_train if idx==0 else ym_train
            rm_train = self.compute_residuals(Y_train, ym_train_1)

            model = DecisionTreeRegressor(
                max_depth=self.max_depth, min_sample_split=self.min_sample_split, min_sample_leaf=self.min_sample_leaf,
                max_features=self.max_features
            )
            logger.info(f'Learning Tree - {idx} | residual {rm_train}')
            model.fit(X=X_train, y=rm_train)
            hm = np.asarray(model.predict(X_train), dtype=np.float64)
            logger.info(f'Tree Predictions - {idx} | predictions {hm}')
            ym_train = self.fitted_tree_prediction(model, X_train, ym_train_1)
            self.training_loss[idx] = self.calculate_loss(Y_train, ym_train)
            
            if X_val is not None and Y_val is not None:
                ym_val_1 = y0_val if idx==0 else ym_val
                ym_val = self.fitted_tree_prediction(model, X_val, ym_val_1)
                logger.debug(f'here are the validation pred | {ym_val}')
                self.validation_loss[idx] = self.calculate_loss(Y_val, ym_val)

            self.random_trees.append(IndividualTree(model=model))

    def predict(self, x:np.ndarray | list) -> np.ndarray:

        x_pred = np.asarray(x)
        predictions = np.full(x_pred.shape[0], self.init_pred)

        for tree in self.random_trees:
            tree_predictions = np.asarray(tree.model.predict(x_pred), dtype=predictions.dtype)
            predictions += self.learning_rate * tree_predictions

        return predictions

    def initial_prediction(self, y: np.ndarray) -> float:
        """
        Optimal constant prediction under
        squared-error loss.
        """

        return float(np.mean(y))

    def compute_residuals(self, y: np.ndarray,predictions: np.ndarray) -> np.ndarray:
        """
        Compute negative gradients for
        half-squared-error loss.
        """
        return y-predictions

    def calculate_loss(self, y : np.ndarray, y_predict : np.ndarray) -> float:

        return float(np.sum((y-y_predict)**2)/y.shape[0])

    def fitted_tree_prediction(self, model : DecisionTreeRegressor, X : np.ndarray, ym_1 : np.ndarray)-> np.ndarray:

        hm = np.asarray(model.predict(X), dtype=np.float64)
        return ym_1 + self.learning_rate*hm