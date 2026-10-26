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
    rng = np.random.default_rng(seed)

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
    rng.shuffle(indices)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    # split arrays
    res = []
    for a in arrays:
        res.append(a[train_idx])
        res.append(a[val_idx])

    return res


def balance_dataset(x, y, negative_class_label=-1, balance_number=6, seed=None):
    """Downsample the dataset to balance positive and negative classes. The resulting dataset tuple will contain (1 / balance_number) of the negative examples and all positive examples.

    Args:
        x: numpy array of features
        y: numpy array of class labels
        negative_class_label (int, defaults to -1): The label of the negative class. Should be either 0 or -1, depending on the prediction method.
        balance_number (int, defaults to 6): The factor of how much the dataset should be balanced. The result will contain (1 / balance_number) of the negative examples.
        seed (int, defaults to None): Controls the shuffling applied to the data. Pass an int for reproducible output across multiple function calls.

    Returns:
        tuple (x, y) of downsampled datasets
    """
    # set seed for reproducibility
    rng = np.random.default_rng(seed)

    # select fraction 1 / balance_number of the negative instances
    y_negative = y[y == negative_class_label]
    x_negative = x[y == negative_class_label]
    negative_class_count = int(y_negative.shape[0] / balance_number)

    permutation = rng.permutation(y_negative.shape[0])
    y_negative = y_negative[permutation]
    x_negative = x_negative[permutation]

    # concatenate positive instances and selected negative instances
    new_x = np.concatenate((x[y == 1], x_negative[:negative_class_count]))
    new_y = np.concatenate((y[y == 1], y_negative[:negative_class_count]))

    # shuffle resulting data
    permutation = rng.permutation(new_y.shape[0])
    return new_x[permutation], new_y[permutation]


def build_poly(x, degree):
    """Polynomial basis function for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,)
        degree: integer

    Returns:
        poly: numpy array of shape (N, degree+1)
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def add_bias(x):
    """Build tx by adding the bias term to x.
    Alias for build_poly(x, 1)

    Args:
        x: numpy array of shape (N,)

    Returns:
        tx: numpy array of shape (N, 2)
    """
    return build_poly(x, 1)
