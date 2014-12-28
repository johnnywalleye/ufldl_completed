import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def KL_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def initialize(hidden_size, visible_size):
    # we'll choose weights uniformly from the interval [-r, r]
    r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.random((hidden_size, visible_size)) * 2 * r - r
    W2 = np.random.random((visible_size, hidden_size)) * 2 * r - r

    b1 = np.zeros(hidden_size, dtype=np.float64)
    b2 = np.zeros(visible_size, dtype=np.float64)

    theta = np.concatenate((W1.reshape(hidden_size * visible_size),
                            W2.reshape(hidden_size * visible_size),
                            b1.reshape(hidden_size),
                            b2.reshape(visible_size)))

    return theta


# visible_size: the number of input units (probably 64)
# hidden_size: the number of hidden units (probably 25)
# lambda_: weight decay parameter
# sparsity_param: The desired average activation for the hidden units (denoted in the lecture
#                            notes by the greek alphabet rho, which looks like a lower-case "p").
# beta: weight of sparsity penalty term
# data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.
#
# The input theta is a vector (because minFunc expects the parameters to be a vector).
# We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
# follows the notation convention of the lecture notes.
# Returns: (cost,gradient) tuple
def sparse_autoencoder_cost_not_vectorized(theta, visible_size, hidden_size,
                            lambda_, sparsity_param, beta, data):
    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    b2 = theta[2 * hidden_size * visible_size + hidden_size:]

    W1grad = np.zeros(W1.shape)
    W2grad = np.zeros(W2.shape)
    b1grad = np.atleast_2d(np.zeros(b1.shape)).T
    b2grad = np.atleast_2d(np.zeros(b2.shape)).T
    num_rows = len(data.T)
    cost = 0.0

    rho_avg = get_rho_avg(data, W1, b1)
    rho_avg_1d = rho_avg[:, 0]
    sparsity_penalty = get_sparsity_penalty(sparsity_param, beta, rho_avg_1d)

    for row_unshaped in data.T:
        row = np.atleast_2d(row_unshaped).T
        z2 = np.dot(W1, row) + np.atleast_2d(b1).T
        a2 = sigmoid(z2)
        z3 = np.dot(W2, a2) + np.atleast_2d(b2).T
        a3 = sigmoid(z3)  # a3 is equivalent to h_Wb in our case

        d3 = -(row - a3) * sigmoid_prime(z3)
        sparsity_param_arr = sparsity_param * np.ones(rho_avg_1d.shape)
        addl_penalty_deriv = beta * (-sparsity_param_arr / rho_avg_1d + (1 - sparsity_param_arr) / (1 - rho_avg_1d))
        addl_penalty_deriv = np.atleast_2d(addl_penalty_deriv).T
        d2 = (np.dot(W2.T, d3) + addl_penalty_deriv) * sigmoid_prime(z2)

        cost += 0.5 * np.power(np.linalg.norm(a3 - row), 2)
        W1grad += np.outer(d2, row.T)
        W2grad += np.outer(d3, a2.T)
        b1grad += d2
        b2grad += d3

    cost /= num_rows
    W1grad /= num_rows
    W2grad /= num_rows
    b1grad /= num_rows
    b2grad /= num_rows

    cost += 0.5 * lambda_ * np.sum(np.power(W1, 2))
    cost += 0.5 * lambda_ * np.sum(np.power(W2, 2))
    W1grad += lambda_ * W1
    W2grad += lambda_ * W2

    cost += sparsity_penalty

    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.
    grad = np.concatenate((W1grad.reshape(hidden_size * visible_size),
                           W2grad.reshape(hidden_size * visible_size),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(visible_size)))

    return cost, grad


def sparse_autoencoder_cost(theta, visible_size, hidden_size,
                                       lambda_, sparsity_param, beta, data):
    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    b2 = theta[2 * hidden_size * visible_size + hidden_size:]

    num_rows = len(data.T)

    rho_avg = get_rho_avg(data, W1, b1)
    rho_avg_1d = rho_avg[:, 0]
    sparsity_penalty = get_sparsity_penalty(sparsity_param, beta, rho_avg_1d)

    z2 = np.dot(W1, data) + np.repeat(np.atleast_2d(b1), num_rows, 0).T
    a2 = sigmoid(z2)
    z3 = np.dot(W2, a2) + np.repeat(np.atleast_2d(b2), num_rows, 0).T
    a3 = sigmoid(z3)

    d3 = -(data - a3) * sigmoid_prime(z3)
    sparsity_param_arr = sparsity_param * np.ones(rho_avg_1d.shape)
    addl_penalty_deriv = beta * (-sparsity_param_arr / rho_avg_1d + (1 - sparsity_param_arr) / (1 - rho_avg_1d))
    addl_penalty_deriv = np.atleast_2d(addl_penalty_deriv).T
    d2 = (np.dot(W2.T, d3) + np.repeat(addl_penalty_deriv, num_rows, 1)) * sigmoid_prime(z2)

    W1grad = np.dot(d2, data.T)
    W2grad = np.dot(d3, a2.T)
    b1grad = d2.sum(axis=1)
    b2grad = d3.sum(axis=1)
    cost = 0.5 * np.sum(np.power(np.linalg.norm(a3 - data, ord=2, axis=0), 2))

    W1grad /= num_rows
    W2grad /= num_rows
    b1grad /= num_rows
    b2grad /= num_rows
    cost /= num_rows

    cost += 0.5 * lambda_ * np.sum(np.power(W1, 2))
    cost += 0.5 * lambda_ * np.sum(np.power(W2, 2))
    W1grad += lambda_ * W1
    W2grad += lambda_ * W2

    cost += sparsity_penalty

    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.
    grad = np.concatenate((W1grad.reshape(hidden_size * visible_size),
                           W2grad.reshape(hidden_size * visible_size),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(visible_size)))

    return cost, grad


def get_rho_avg(data, W1, b1):
    rho_total = np.zeros(np.atleast_2d(b1).T.shape)
    for row_unshaped in data.T:
        row = np.atleast_2d(row_unshaped).T
        z2 = np.dot(W1, row) + np.atleast_2d(b1).T
        a2 = sigmoid(z2)
        rho_total += a2
    num_rows = len(data.T)
    return rho_total / num_rows

def get_sparsity_penalty(sparsity_param, beta, rho_avg_1d):
    kl_divergence = KL_divergence(sparsity_param * np.ones(rho_avg_1d.shape), rho_avg_1d)
    sparsity_penalty = np.sum(kl_divergence)
    return beta * sparsity_penalty