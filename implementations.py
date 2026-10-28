import numpy as np
import random
import matplotlib.pyplot as plt
import time

from evals import accuracy_from_model


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


def compute_mean_squared_error_gradient(y, tx, w):
    """Computes the linear regression gradient at w.

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
        grad = compute_mean_squared_error_gradient(y, tx, w)
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
        xn = tx[n : n + 1]
        yn = y[n : n + 1]
        grad = compute_mean_squared_error_gradient(yn, xn, w)
        w = w - gamma * grad

    # compute MSE loss for the last weight vector
    loss = mean_squared_error(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y (np.ndarray): target output of shape=(N,)
        tx (np.ndarray): input data of shape=(N, D)

    Returns:
        (w, loss): the optimal weight vector of shape (D,), and the corresponding MSE loss value.
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = mean_squared_error(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """
    D = tx.shape[1]
    N = tx.shape[0]

    a = tx.T @ tx + 2 * N * lambda_ * np.identity(D)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp(-np.clip(t, -500, 500)))


def negative_log_likelihood(y, tx, w):
    """Compute the loss by negative log likelihood.
    This does not include the penalty term, even for regularized_logistic_regression.

    Args:
        y: shape=(N, 1)
        tx: shape=(N, D)
        w: shape=(D, 1)

    Returns:
        loss, a non-negative scalar
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    # -y * tx @ w + log(1 + exp(tx @ w))
    z = tx @ w
    return np.mean(-y * z + np.logaddexp(0, z))


def logistic_regression_gradient(y, tx, w, lambda_=0):
    """Compute the gradient for logistic regression with optional regularization term.

    Args:
        y: shape=(N, 1)
        tx: shape=(N, D)
        w: shape=(D, 1)
        lambda_: scalar, regularization parameter (default: 0)

    Returns:
        gradient: shape=(D, 1)
    """
    N = len(y)
    p = sigmoid(tx @ w)
    return (1 / N) * (tx.T @ (p - y)) + 2 * lambda_ * w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Do max_iters steps of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1)
        max_iters (int): total number of SGD iterations
        gamma: float

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Do max_iters steps of gradient descent using logistic regression with L2 regularization. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda_: scalar
        intial_w:  shape=(D, 1)
        max_iters (int): total number of SGD iterations
        gamma: float

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == initial_w.shape[0]

    w = initial_w.copy()

    for _ in range(max_iters):
        grad = logistic_regression_gradient(y, tx, w, lambda_)
        w = w - gamma * grad

    # Unregularized negative log likelihood
    loss = negative_log_likelihood(y, tx, w)

    return w, loss


def calculate_primal_objective(y, X, w, lambda_):
    """compute the full cost (the primal objective, equation (1) in the exercise pdf),
        that is loss plus regularizer.

    Args:
        X: the full dataset matrix, shape = (num_examples, num_features)
        y: the corresponding +1 or -1 labels, shape = (num_examples)
        w: shape = (num_features)

    Returns:
        scalar, non-negative
    """
    return (
        np.mean(np.clip(1 - y * (X @ w), 0, np.inf)) + (lambda_ / 2) * np.sum((w**2))
    ).tolist()


def calculate_stochastic_gradient_svm(y, X, w, lambda_, n, num_examples):
    """compute the stochastic gradient of loss plus regularizer.

    Args:
        X: the dataset matrix, shape = (num_examples, num_features)
        y: the corresponding +1 or -1 labels, shape = (num_examples)
        w: shape = (num_features)
        lambda_: positive scalar number
        n: the index of the (one) datapoint we have sampled
        num_examples: N

    Returns:
        numpy array, shape = (num_features)
    """
    x_n = X[n]
    y_n = y[n]
    if y_n * (x_n @ w) > 1:
        return lambda_ * w
    return -y_n * x_n + lambda_ * w


def sgd_for_svm_demo(
    y, X, max_iter: int = 1000, gamma: float = 1e-4, lambda_: float = 5e1
):
    xs = np.unique(np.round(np.logspace(0, 5, 201)))[:-1]
    num_examples, num_features = X.shape
    w = np.zeros(num_features)
    costs = []
    start = time.time()

    for it in range(max_iter):
        # n = sample one data point uniformly at random data from x
        n = random.randint(0, num_examples - 1)

        grad = calculate_stochastic_gradient_svm(y, X, w, lambda_, n, num_examples)
        w -= gamma / (it + 1) * grad

        cost = calculate_primal_objective(y, X, w, lambda_)

        if it in xs:
            costs += [cost]

    end = time.time()

    print(
        "Final training accuracy = {l} %".format(
            l=np.round(100 * accuracy_from_model(y, X, w), 2)
        )
    )
    print("Training time: " + str(np.round(end - start, 1)) + " seconds ")

    plt.figure()
    plt.title("SGD on primal objective")
    if len(xs) > len(costs):
        xs = xs[: len(costs)]
    plt.loglog(xs, costs)
    plt.xlabel("Number of iterations")
    plt.ylabel("Primal objective")
    plt.show()
    return costs, w
