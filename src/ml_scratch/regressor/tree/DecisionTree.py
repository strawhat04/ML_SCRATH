import numpy as np
from loguru import logger
from ml_scratch.regressor.splitter.best_splitter import BestFitter 
from ml_scratch.classifier.utils.validator import validate_array
from ml_scratch.regressor.utils.models import Node, bestsplitnode
from ml_scratch.regressor.criterion.sse import squared_err

class DecisionTreeRegressor:

    def __init__(self, max_depth: int, min_sample_split: int, min_sample_leaf: int, max_features : None | int) -> None:
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.max_features = max_features

        self.xtrain = None
        self.ytrain = None
        self.y_pred = None
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Node:
        """
        Fit the decision tree classifier on the input training data. 

            Recursively builds the decision tree starting from the root node using
            the provided feature matrix and target labels.

            Args:
                X (np.ndarray): Training feature matrix of shape (n_samples, n_features).
                y (np.ndarray): Target class labels of shape (n_samples,).

            Returns:
                DecisionTreeRegressor: The fitted tree instance (self).

            Raises:
                ValueError: If `X` and `y` have incompatible shapes or if `X` is empty.
        """
        validate_array('Feature', X)
        validate_array('Label', y)

        logger.info(
            f"Starting tree construction | Samples: {X.shape[0]}, Features: {X.shape[1]}, "
            f"Max Depth: {self.max_depth}"
        )

        self.xtrain = X
        self.ytrain = y

        self.root = self._build_tree(X, y, depth=0)
        
        logger.success("Tree fitting completed successfully!")
        return self.root

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node :
        """Recursively build the decision tree by finding optimal feature splits.

        At each step, evaluates candidate splits to minimize variance (sse index).
        Recursion stops and a leaf node is created if stopping criteria are met
        (e.g., reaching `max_depth`, node purity, or falling below `min_samples_split`).

        Args:
            X (np.ndarray): Feature matrix for the current node subset of shape
                (n_node_samples, n_features).
            y (np.ndarray): Target class labels for the current node subset of shape
                (n_node_samples,).
            depth (int, optional): Current recursion depth level within the tree.
                Defaults to 0.

        Returns:
            Node: Root node of the constructed subtree (either a decision node with
            left/right children or a leaf node containing a class prediction).
        """
        indent = "  " * depth  # Visual hierarchy in terminal

        stop, reason = self._stopping_criteria(y, depth)
        
        if stop:
            logger.debug(f'Stopping Criteria met | reason = {reason}')
            leaf_value = self._get_prediction(y)
            current_sse = squared_err(y)
            logger.debug(
                f"{indent} Leaf Node [Depth {depth}] | Reason: {reason} | "
                f"Samples: {y.size} | sse: {current_sse:.4f} | Pred: {leaf_value}"
            )
            return Node(
                id=depth,
                feature_index=None,
                split_threshold=None,
                sse=current_sse,
                n_samples=y.size,
                prediction=leaf_value,
                left=None,
                right=None
            )


        if self.max_features is None or self.max_features > X.shape[1]:
            logger.info(f"max_features is None or exceeds feature count; using all {X.shape[1]} features for split evaluation.")
            max_features = X.shape[1]
        else:
            max_features = self.max_features
            logger.info(f"Selecting {max_features} features randomly for split evaluation at depth {depth}.")

        feature_best_node = np.empty(max_features, dtype=object)
        features_best_sse = np.empty(max_features)

        allowed_features = np.random.choice(X.shape[1], size=max_features, replace=False)
        logger.info(
           f"Depth {depth}: selected features {allowed_features}"
        )

        for idx, feat_idx in enumerate(allowed_features):
            bestfit = BestFitter(X[:, feat_idx], y, self.min_sample_leaf)
            feature_best_node[idx] = bestfit.find_best_split()
            features_best_sse[idx] = feature_best_node[idx].sse

        best_node, local_feat_index = self._get_best_feature_node(feature_best_node, features_best_sse)
        feature_index = allowed_features[local_feat_index] if best_node is not None else None

        if best_node is not None:
            logger.info(f"{indent} Depth {depth}: Best split found on feature {feature_index} with threshold {best_node.value:.4f} and sse {best_node.sse:.4f}")

        if best_node is None:
            leaf_value = self._get_prediction(y)
            current_sse = squared_err(y)
            logger.debug(
                f"{indent} Leaf Node [Depth {depth}] | Reason: No further Split Exist | "
                f"Samples: {y.size} | sse: {current_sse:.4f} | Pred: {leaf_value}"
            )
            return Node(
                id=depth,
                feature_index=None,
                split_threshold=None,
                sse=current_sse,
                n_samples=y.size,
                prediction=leaf_value,
                left=None,
                right=None
            )

        logger.info(
            f"{indent} Split [Depth {depth}] | Feature {feature_index} < {best_node.value:.4f} | "
            f"sse: {best_node.sse:.4f} | Samples: {y.size}"
        )

        left_mask = X[:, feature_index] < best_node.value
        right_mask = ~left_mask

        Xleft, yleft = X[left_mask], y[left_mask]
        Xright, yright = X[right_mask], y[right_mask]

        left_child = self._build_tree(Xleft, yleft, depth=depth + 1)
        right_child = self._build_tree(Xright, yright, depth=depth + 1)

        return Node(
            id=depth,
            feature_index=feature_index,
            split_threshold=best_node.value,
            sse=best_node.sse,
            n_samples=y.size,
            prediction=None,
            left=left_child,
            right=right_child,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Traverse the decision tree using boolean masks to predict target labels.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).

        Raises:
            ValueError: If called before model is fitted (`self.root` is None).
        """
        if self.root is None:
            raise ValueError("Run fit() before prediction")

        logger.info(f"Starting prediction for dataset with shape {X.shape}.")

        self.y_pred = np.full(X.shape[0], fill_value=None, dtype=object)
        idx = np.full(X.shape[0], fill_value=True)

        predictions = self._get_depth_leaf(X, idx, node=self.root, depth=0)
        logger.success("Prediction completed successfully.")
        return predictions

    def _get_depth_leaf(self, X: np.ndarray, idx: np.ndarray, node: Node, depth: int) -> np.ndarray:
        """Recursively traverse nodes to assign leaf predictions using boolean indices.

        Args:
            X (np.ndarray): Input feature matrix.
            idx (np.ndarray): Boolean mask indicating active samples at current node.
            node (Node): Current tree node being evaluated.
            depth (int): Current recursion depth level.

        Returns:
            np.ndarray: Updated array of predicted labels `self.y_pred`.
        """
        active_count = np.sum(idx)
        indent = "  " * depth

        logger.debug(f"{indent}[Depth {depth}] Node processing {active_count} active samples.")

        if depth > self.max_depth:
            logger.warning(
                f"{indent}[Depth {depth}] Exceeded max_depth ({self.max_depth}). Stopping branch traversal."
            )
            return self.y_pred

        if node.prediction is not None:
            self.y_pred[idx] = node.prediction
            logger.debug(
                f"{indent}[Depth {depth}] Leaf node hit! Assigned prediction={node.prediction} "
                f"to {active_count} samples."
            )

        else:
            logger.debug(
                f"{indent}[Depth {depth}] Decision Split: Feature {node.feature_index} < {node.split_threshold:.4f}"
            )

            left_id = (X[:, node.feature_index] < node.split_threshold) & idx
            right_id = (X[:, node.feature_index] >= node.split_threshold) & idx

            logger.debug(
                f"{indent}[Depth {depth}] Branch routing: {np.sum(left_id)} samples -> Left, "
                f"{np.sum(right_id)} samples -> Right"
            )

            left_node = node.left
            right_node = node.right

            if left_node is not None:
                logger.debug(f"{indent}[Depth {depth}] Traversing Left Child...")
                self._get_depth_leaf(X, idx=left_id, node=left_node, depth=depth + 1)

            if right_node is not None:
                logger.debug(f"{indent}[Depth {depth}] Traversing Right Child...")
                self._get_depth_leaf(X, idx=right_id, node=right_node, depth=depth + 1)

        return self.y_pred


    def _stopping_criteria(self, y: np.ndarray, depth: int) -> tuple[bool, str]:
        if squared_err(y) == 0:
            return True, "Pure Node (sse=0)"
        if depth >= self.max_depth:
            return True, f"Max Depth Reached ({depth}>={self.max_depth})"
        if y.size < self.min_sample_split:
            return True, f"Min Samples Split Reached ({y.size}<{self.min_sample_split})"

        return False, ""

    def _get_best_feature_node(self, nodes: np.ndarray, sse: np.ndarray) -> tuple[bestsplitnode | None, int]:
        if np.all(sse==np.inf):
            return None, -1
        min_idx = sse.argmin()
        return nodes[min_idx], int(min_idx)
    
    def _get_prediction(self, y: np.ndarray) -> float:
        return float(np.mean(y))

    @staticmethod
    def print_tree(node: Node, feature_names: list[str] | None = None, depth: int = 0) -> None:
        """
        Recursively print a decision tree in indented text form.

        Args:
            node: root (or current) Node to print.
            feature_names: list where feature_names[i] is the display name
                            for feature index i. If None, falls back to
                            "feature_<index>".
            depth: current recursion depth, controls indentation (internal use).
        """
        indent = "  " * depth

        if node.left is None and node.right is None:
            print(f"{indent}Leaf -> prediction={node.prediction} "
                f"(sse={node.sse:.4f}, samples={node.n_samples})")
            return

        name = (feature_names[node.feature_index]
                if feature_names is not None
                else f"feature_{node.feature_index}")

        print(f"{indent}[{name} <= {node.split_threshold:.4f}] "
            f"(sse={node.sse:.4f}, samples={node.n_samples})")

        print(f"{indent}├─ True:")
        DecisionTreeRegressor.print_tree(node.left, feature_names, depth + 1)

        print(f"{indent}└─ False:")
        DecisionTreeRegressor.print_tree(node.right, feature_names, depth + 1)

    def get_tree_structure(self) -> dict:
        """
        Retrieve the structure of the decision tree as a nested dictionary.

        Returns:
            dict: A nested dictionary representing the tree structure.
        """
        if self.root is None:
            logger.warning("No tree has been fitted yet. Returning empty structure.")
            return {}

        def _node_to_dict(node: Node) -> dict:
            if node is None:
                return None

            node_dict = {
                "id": node.id,
                "feature_index": node.feature_index,
                "split_threshold": node.split_threshold,
                "sse": node.sse,
                "n_samples": node.n_samples,
                "prediction": node.prediction,
                "left": _node_to_dict(node.left),
                "right": _node_to_dict(node.right)
            }
            return node_dict

        logger.info("Retrieving structure of the decision tree.")
        return _node_to_dict(self.root)

    def feature_importance(self) -> np.ndarray:
        """
        Calculate the feature importance based on the sse variance reduction.

        Returns:
            np.ndarray: An array of feature importance scores, where each score
                        corresponds to a feature index.
        """
        if self.root is None or self.xtrain is None :
            logger.warning("No tree has been fitted yet. Returning empty feature importance.")
            return np.array([])

        importance_scores = np.zeros(self.xtrain.shape[1])
        self._accumulate_importance(self.root, importance_scores)

        return importance_scores 
    
    def _accumulate_importance(self, node: Node, importance_scores : np.ndarray, depth : int = 0) -> None:
        if node is None:
            return None

        if node.left is None and node.right is None:
            return None
    
        feature_idx = node.feature_index
        n_samples = node.n_samples
        parent_sse = node.sse

        child_sse = node.left.sse * node.left.n_samples + node.right.sse * node.right.n_samples
        sse_reduction = parent_sse * n_samples - child_sse

        logger.debug(
            f"[Depth {depth}] Feature {feature_idx} | sse Reduction: {sse_reduction:.4f} | "
            f"Samples: {n_samples} | Parent sse: {parent_sse:.4f} | "
            f"Left sse: {node.left.sse:.4f}, Right sse: {node.right.sse:.4f}"
        )

        importance_scores[feature_idx] += sse_reduction

        self._accumulate_importance(node.left, importance_scores, depth + 1)
        self._accumulate_importance(node.right, importance_scores, depth + 1)