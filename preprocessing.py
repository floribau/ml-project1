import numpy as np


def train_val_split(*arrays, val_ratio=0.25, seed=None):
    """Split arrays into random train and validation subsets.

    Args:
        *arrays (sequence of array-like of shape (n_samples,) or (n_samples, n_outputs)): Data to be split with consistent first dimension.
        val_ratio (float, default=0.25): Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split.
        seed (int, default=None): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

    Returns:
        list, length=2 * len(arrays): List containing train-validation split of inputs.
    """
    # set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # check input arrays
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    # convert inputs to numpy arrays if they aren't already
    arrays = [np.asarray(a) for a in arrays]

    # validate that all arrays have the same row length
    n_samples = arrays[0].shape[0]
    for i, a in enumerate(arrays[1:], start=1):
        if a.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent number of samples: array 0 has {n_samples}, but array {i} has {a.shape[0]}"
            )

    # compute validation set size
    val_size = int(n_samples * val_ratio)

    # shuffle and distribute indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    # split arrays
    res = []
    for a in arrays:
        res.append(a[train_idx])
        res.append(a[val_idx])

    return res
