import numpy as np
from dataclasses import dataclass
from loguru import logger
from ml_scratch.classifier.splitter.candidate_splits import candidate_splits
from ml_scratch.regressor.criterion.sse import squared_err
from ml_scratch.regressor.utils.models import bestsplitnode
from ml_scratch.classifier.utils.validator import validate_array

@dataclass
class BestFitter:

    feature : np.ndarray
    label : np.ndarray
    min_leaf_size : int


    def __post_init__(self):
        # Initial validation checks
        validate_array('feature', self.feature)
        validate_array('label', self.label)
        
        if self.feature.size != self.label.size:
            logger.error(f"Validation failed: Size mismatch (feature={self.feature.size}, label={self.label.size})")
            raise ValueError(
                f'label and feature size mismatch | feature size : {self.feature.size} , label size : {self.label.size}'
            )

        # Edge-case warning for small dataset splits
        if self.feature.size < 2:
            logger.warning(f"Dataset is extremely small (n={self.feature.size}). Splitting may be unreliable.")

    def find_best_split(self) -> bestsplitnode:

        candidate = candidate_splits(feature=self.feature)

        candidate_sse = np.full(candidate.size, np.inf)
        total_samples = self.feature.size

        logger.info(f"Evaluating {candidate.size} potential candidate splits across {total_samples} samples...")

        for idx, potential in enumerate(candidate):
            split_left = self.label[self.feature < potential]
            split_right = self.label[self.feature >= potential]

            left_size = split_left.size
            right_size = split_right.size

            # Skip the split if failing min_sample criteria
            if left_size <self.min_leaf_size or right_size<self.min_leaf_size:
                logger.warning(
                    f"Skipping [Split {idx+1:02d}/{candidate.size:02d}] Threshold {potential:<8.4f} "
                    f"resulted in low sample leaf! (Left: {left_size}, Right: {right_size})"
                )
                continue

            node_left_sse = squared_err(split_left)
            node_right_sse = squared_err(split_right)

            weighted_sse = node_left_sse + node_right_sse
            candidate_sse[idx] = weighted_sse

            # Lean, visual single-line log per iteration
            logger.debug(
                f"[Split {idx+1:02d}/{candidate.size:02d}] "
                f"Threshold: {potential:<8.4f} | "
                f"sse: {weighted_sse:.4f} | "
                f"Left (n={left_size}): {node_left_sse:.4f} | "
                f"Right (n={right_size}): {node_right_sse:.4f}"
            )

        best_split_node = self._get_minimum_sse_node(candidate_sse, candidate)
        return best_split_node
    
    def _get_minimum_sse_node(self, candidate_sse: np.ndarray, candidate_node: np.ndarray) -> bestsplitnode:
        """
        Find the minimum sse score and corresponding node split.
        """

        if candidate_sse.size == 0 or np.all(np.isinf(candidate_sse)):
            logger.warning("No valid splits found. All candidate splits were invalid or resulted in infinite sse.")
            return bestsplitnode(sse=np.inf, value=np.nan)

        min_idx = candidate_sse.argmin()
        min_sse = candidate_sse[min_idx].item()
        node_value = candidate_node[min_idx].item()

        # Highlight optimal result with SUCCESS level
        logger.success(
            f"Best Split Selected -> Threshold: {node_value:.4f} | Minimum sse: {min_sse:.4f} (Index: {min_idx})"
        )

        return bestsplitnode(sse=min_sse, value=node_value)