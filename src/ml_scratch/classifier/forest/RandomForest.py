import numpy as np
from loguru import logger
from dataclasses import dataclass, field

from ml_scratch.classifier.utils.models import Node
from ml_scratch.classifier.tree.DecisionTree import DecisionTreeClassifier

@dataclass
class IndvidualTree:
    trees : Node | None = None 
    model : DecisionTreeClassifier | None = None
    oob_index : np.ndarray | None = None


@dataclass
class RandomForest:

    n_estimators: int = 1
    max_depth: int = 5
    min_sample_leaf: int = 1
    min_sample_split: int = 2
    max_features: int | None = None
    random_trees: list[IndvidualTree] = field(default_factory=list)
    oob_score_: float | None = None

    def _bootstrap_sample(self, n_samples: int) -> np.ndarray:
        random_idx = np.random.randint(0, n_samples, n_samples)
        logger.debug(f"Random indices for bootstrap sample: {random_idx}")
        return random_idx

    
    def fit(self, X: np.ndarray | list, y: np.ndarray | list) -> None:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must contain the same number of observations")

        n_samples = len(y_arr)
        self.random_trees = []
        oob_votes: list[list] = [[] for _ in range(n_samples)]

        logger.info("Fitting Random Forest model")
        for i in range(self.n_estimators):
            logger.info(f"Training tree: {i + 1}/{self.n_estimators}")
            sample_idx = self._bootstrap_sample(n_samples)
            X_sample, y_sample = X_arr[sample_idx], y_arr[sample_idx]
            X_oob, _, oob_indices = self._retain_oob_samples(X_arr, y_arr, sample_idx)

            model = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_sample_leaf=self.min_sample_leaf,
                min_sample_split=self.min_sample_split,
                max_features=self.max_features,
            )
            tree_root = model.fit(X_sample, y_sample)

            if len(oob_indices) > 0:
                preds = model.predict(X_oob)
                for idx, pred in zip(oob_indices, preds):
                    oob_votes[idx].append(pred)

            self.random_trees.append(IndividualTree(tree=tree_root, model=model))

        self.oob_score_ = self._calculate_forest_oob(y_arr, oob_votes)
        logger.info(f"Forest fitting complete | OOB Score: {self.oob_score_:.4f}")
            
    def predict(self, X: np.ndarray | list) -> np.ndarray:
        logger.info("Predicting using Random Forest model")
        x_arr = np.asarray(X)
        all_prediction = np.array([tree.model.predict(x_arr) for tree in self.random_trees])

        return self._ensemble_predictions(all_prediction)
    
    def _ensemble_predictions(self, all_predictions: np.ndarray) -> np.ndarray:
        logger.info("Ensembling predictions from all trees")
        pred_pre_trees = all_predictions.T.astype(int)
        return np.array([np.bincount(each_pred).argmax() for each_pred in pred_pre_trees])

    def _get_tree_structure(self) -> dict:
        if not self.random_trees:
            logger.warning("No trees have been fitted yet. Returning empty structure.")
            return {}

        logger.info("Retrieving structure of all trees in the Random Forest")
        return {f"Tree_{i}": tree.model.get_tree_structure() for i, tree in enumerate(self.random_trees)}

    def _retain_oob_samples(
        self, X: np.ndarray, y: np.ndarray, sample_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = len(y)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[np.unique(sample_indices)] = False
        oob_indices = np.where(oob_mask)[0]

        return X[oob_mask], y[oob_mask], oob_indices

    def _calculate_forest_oob(self, y_true: np.ndarray, oob_votes: list[list]) -> float:
        logger.info("Calculating aggregated Out-Of-Bag (OOB) score")

        evaluated_indices = []
        final_preds = []

        for i, votes in enumerate(oob_votes):
            if len(votes) == 0:
                continue
            vals, counts = np.unique(votes, return_counts=True)
            final_preds.append(vals[counts.argmax()])
            evaluated_indices.append(i)

        if not evaluated_indices:
            logger.warning("No samples were out-of-bag across estimators. OOB score is undefined.")
            return 0.0

        return float(np.mean(y_true[evaluated_indices] == np.array(final_preds)))

        