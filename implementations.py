import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
import time

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
    return (np.mean(np.clip(1 - y * (X@w), 0, np.inf)) + (lambda_ /2) * np.sum((w**2))).tolist()

def calculate_accuracy(y, X, w):
    """compute the accuracy on the given dataset (X, y) using the model w.

    Args:
        X: the full dataset matrix, shape = (num_examples, num_features)
        y: the corresponding +1 or -1 labels, shape = (num_examples)
        w: shape = (num_features)

    Returns:
        scalar, between 0 and 1
    """
    correct_prediction = y * (X@w) # prediction is correct if the sign is positive.
    return correct_prediction[correct_prediction >= 0].shape[0] / y.shape[0]

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
    return -y_n * x_n + lambda_*w

def sgd_for_svm_demo(y, X, max_iter: int = 1000, gamma:float = 1e-4, lambda_:float = 5e1):
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
            l=np.round(100 * calculate_accuracy(y, X, w), 2)
        )
    )
    print("Training time: " + str(np.round(end - start, 1)) + " seconds ")

    plt.figure()
    plt.title("SGD on primal objective")
    if len(xs) > len(costs):
        xs = xs[:len(costs)]
    plt.loglog(xs, costs)
    plt.xlabel("Number of iterations")
    plt.ylabel("Primal objective")
    plt.show()
    return costs, w