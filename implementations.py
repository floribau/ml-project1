import numpy as np

def mean_squared_error(y, tx, w):
    """Calculate the MSE loss.

    Args:
        y (np.ndarray): target output of shape=(N,)
        tx (np.ndarray): input data of shape=(N, D)
        w (np.ndarray): model parameters of shape=(D,)

    Returns:
        The scalar value of MSE loss corresponding to the input parameters w.
    """
    e = y - tx @ w
    # factor 0.5 to be consistent with the lecture notes
    return 0.5 * np.mean(e**2)


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y (np.ndarray): target output of shape=(N,)
        tx (np.ndarray): input data of shape=(N, D)
        w (np.ndarray): model parameters of shape=(D,)

    Returns:
        The gradient of the loss at w.
    """
    e = y - tx @ w
    N = len(e)
    return -(tx.T @ e) / N


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent (GD).

    Args:
        y (np.ndarray): target output of shape=(N,)
        tx (np.ndarray): input data of shape=(N, D)
        initial_w (np.ndarray): initial model parameters of shape=(D,)
        max_iters (int): total number of GD iterations
        gamma (float): stepsize

    Returns:
        (w, loss): the weight vector after max_iters GD iterations, and the corresponding MSE loss value.
    """
    w = initial_w.copy()

    # perform max_iters GD iterations
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad

    # compute MSE loss for the last weight vector
    loss = mean_squared_error(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (SGD) with mini batch size 1.
    Set np.random.seed before the method call if deterministic reproducibility is required.

    Args:
        y (np.ndarray): target output of shape=(N,)
        tx (np.ndarray): input data of shape=(N, D)
        initial_w (np.ndarray): initial model parameters of shape=(D,)
        max_iters (int): total number of SGD iterations
        gamma (float): stepsize

    Returns:
        (w, loss): the weight vector after max_iters SGD iterations, and the corresponding MSE loss value.
    """
    w = initial_w.copy()
    N = len(y)

    # perform max_iters SGD iterations
    for _ in range(max_iters):
        # randomly pick single datapoint
        n = np.random.randint(N)
        xn = tx[n:n+1]
        yn = y[n:n+1]
        grad = compute_gradient(yn, xn, w)
        w = w - gamma * grad

    # compute MSE loss for the last weight vector
    loss = mean_squared_error(y, tx, w)
    return w, loss

