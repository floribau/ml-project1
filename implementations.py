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

    w = np.linalg.solve(
        (np.matmul(np.transpose(tx), tx) + np.multiply(lambda_*(2*N), np.diag(np.full(D,1)))),
                  np.matmul(np.transpose(tx), y)
    )
    return w

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1/(1+np.exp(-t))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1)
        max_iters (int): total number of SGD iterations
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == initial_w.shape[0]

    w = initial_w.copy()
    N = len(y)

    for _ in range(max_iters):
        gradient = (1/N)*np.matmul(tx.T, sigmoid(np.matmul(tx, w))-y)
        w = w - gamma * gradient

    loss = (-1/N)*np.sum(y*np.log(sigmoid(np.matmul(tx, w))) + (1-y)*np.log(1-sigmoid(np.matmul(tx, w))))

    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda_: scalar
        intial_w:  shape=(D, 1)
        max_iters (int): total number of SGD iterations
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == initial_w.shape[0]

    w = initial_w.copy()
    N = len(y)
    

    for _ in range(max_iters):
        gradient = (1/N)*np.matmul(tx.T, sigmoid(np.matmul(tx, w))-y) + lambda_*2*w
        w = w - gamma * gradient

    loss = (-1/N)*np.sum(y*np.log(sigmoid(np.matmul(tx, w))) + (1-y)*np.log(1-sigmoid(np.matmul(tx, w))))

    return loss, w