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
def sparse_autoencoder_cost(theta, visible_size, hidden_size,
                            lambda_, sparsity_param, beta, data):
    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    b2 = theta[2 * hidden_size * visible_size + hidden_size:]
    
    # SR-71: YOUR CODE GOES HERE
    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.
    grad = np.concatenate((W1grad.reshape(hidden_size * visible_size),
                           W2grad.reshape(hidden_size * visible_size),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(visible_size)))

    return cost, grad


# SR-71: predictions?
def sparse_autoencoder(theta, hidden_size, visible_size, data):
    """
    :param theta: trained weights from the autoencoder
    :param hidden_size: the number of hidden units (probably 25)
    :param visible_size: the number of input units (probably 64)
    :param data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.
    """

    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.
    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]

    # Number of training examples
    m = data.shape[1]

    # Forward propagation
    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)

    return a2




