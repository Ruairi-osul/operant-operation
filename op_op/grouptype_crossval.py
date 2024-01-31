from sklearn.model_selection import BaseCrossValidator
import numpy as np

class GroupTypeKFold(BaseCrossValidator):
    """
    Split grouped data data into k folds, ensuring that each fold contains equal proportions of each group type.

    Groups ~ trials.
    Group types ~ trial types.

    Args:
        n_splits (int): Number of folds. Must be at least 2.
    """

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None, group_types=None):
        if groups is None or group_types is None:
            raise ValueError(
                "The 'groups' and 'group_types' parameters must not be None"
            )

        groups = np.array(groups)
        group_types = np.array(group_types)

        _ = np.unique(groups)
        unique_group_types = np.unique(group_types)

        fold_assignments = np.full(len(X), -1)

        for group_type in unique_group_types:
            type_indices = np.flatnonzero(group_types == group_type)
            type_groups = np.unique(groups[type_indices])

            fold = 0
            for group in type_groups:
                group_indices = np.flatnonzero(groups == group)
                fold_assignments[group_indices] = fold
                fold = (fold + 1) % self.n_splits

        for fold in range(self.n_splits):
            test_idx = np.flatnonzero(fold_assignments == fold)
            train_idx = np.flatnonzero(fold_assignments != fold)
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None, group_types=None):
        return self.n_splits