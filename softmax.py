import numpy as np
import scipy.optimize
import scipy.sparse


def softmax_cost(theta, num_classes, input_size, lambda_, data, labels):
    # theta: theta (
    # num_classes: the number of classes
    # input_size - the size N of each training example
    # lambda_ - weight decay parameter
    # data - the N x M input matrix, where each column data(:, i) corresponds to
    #        a single test set
    # labels - an M x 1 matrix containing the labels corresponding for the input data

    # Unroll the parameters from theta

    # ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Compute the cost and gradient for softmax regression.
    #                You need to compute thetagrad and cost.
    #                The groundTruth matrix might come in handy.
    theta = theta.reshape(num_classes, input_size)
    num_cases = data.shape[1]

    ground_truth = np.array(scipy.sparse.csr_matrix(
        (np.ones(num_cases), (range(num_cases), labels - 1))).todense())
    theta_dot_x = np.dot(data.T, theta.T)
    # class_probabilities = np.log(np.exp(theta_dot_x) /
    #                  np.atleast_2d(np.exp(theta_dot_x).sum(axis=1)).T)
    class_probabilities = np.exp(theta_dot_x) / np.atleast_2d(np.exp(theta_dot_x).sum(axis=1)).T
    overall_prod = np.multiply(ground_truth, class_probabilities)
    overall_prod_y_gt_0 = overall_prod[overall_prod > 0]
    cost = (-1 / num_cases) * np.sum(np.log(overall_prod_y_gt_0))
    cost += (lambda_ / 2) * np.sum(theta ** 2)

    # theta_grad = np.zeros([num_classes, input_size])
    theta_grad = (-1 / num_cases) * np.dot(data, ground_truth - class_probabilities).T
    theta_grad += lambda_ * theta

    # ------------------------------------------------------------------
    # Unroll the gradient matrices into a vector for minFunc
    return cost, theta_grad.ravel()


def softmax_train(input_size, num_classes, lambda_, data, labels, options=None):
    #softmaxTrain Train a softmax model with the given parameters on the given
    # data. Returns softmaxOptTheta, a vector containing the trained parameters
    # for the model.
    #
    # input_size: the size of an input vector x^(i)
    # num_classes: the number of classes
    # lambda_: weight decay parameter
    # input_data: an N by M matrix containing the input data, such that
    #            inputData(:, c) is the cth input
    # labels: M by 1 matrix containing the class labels for the
    #            corresponding inputs. labels(c) is the class label for
    #            the cth input
    # options (optional): options
    #   options.maxIter: number of iterations to train for

    if options is None:
        options = {'maxiter': 400, 'disp': True}

    # Initialize theta randomly
    theta = 0.005 * np.random.randn(num_classes * input_size)

    J = lambda x: softmax_cost(x, num_classes, input_size, lambda_, data, labels)

    result = scipy.optimize.minimize(J, theta, method='l-bfgs-b', jac=True, options=options)

    # Return optimum theta, input size & num classes
    opt_theta = result.x

    return opt_theta, input_size, num_classes


def softmax_predict(softmax_model, data):
    # softmax_model - model trained using softmax_train (opt_theta, input_size, num_classes)
    # data - the N x M input matrix, where each column data(:, i) corresponds to
    #        a single test set
    #
    # Your code should produce the prediction matrix 
    # pred, where pred(i) is argmax_c P(y(c) | x(i)).
     
    # Unroll the parameters from theta
    # theta = softmaxModel.optTheta;  # this provides a num_classes x input_size matrix
    # pred = zeros(1, size(data, 2));
    
    # ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Compute pred using theta assuming that the labels start 
    #                from 1.
    theta = softmax_model[0]
    num_labels = softmax_model[2]
    theta_for_preds = theta.reshape([num_labels, theta.shape[0] / num_labels])
    theta_dot_x = np.dot(data.T, theta_for_preds.T)
    class_probabilities = np.exp(theta_dot_x) / (np.exp(theta_dot_x).sum(axis=1)[:, np.newaxis])
    class_predictions = class_probabilities.argmax(axis=1) + 1
    return class_predictions
