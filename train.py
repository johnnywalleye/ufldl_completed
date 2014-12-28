import numpy as np
import scipy.optimize
import sparse_autoencoder
import gradient
import display_network

def train(visible_size, hidden_size, sparsity_param, lambda_, beta, debug, patches,
          use_vectorized_implementation=False):
    #  Obtain random parameters theta
    theta = sparse_autoencoder.initialize(hidden_size, visible_size)
    if not use_vectorized_implementation:
        sparse_autoencoder_cost_fn = sparse_autoencoder.sparse_autoencoder_cost
    else:
        sparse_autoencoder_cost_fn = sparse_autoencoder.sparse_autoencoder_cost_vectorized

    # =====================================================================
    # STEP 2: Implement sparseAutoencoderCost
    #
    #  You can implement all of the components (squared error cost, weight decay term,
    #  sparsity penalty) in the cost function at once, but it may be easier to do
    #  it step-by-step and run gradient checking (see STEP 3) after each step.  We
    #  suggest implementing the sparseAutoencoderCost function using the following steps:
    #
    #  (a) Implement forward propagation in your neural network, and implement the
    #      squared error term of the cost function.  Implement backpropagation to
    #      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking
    #      to verify that the calculations corresponding to the squared error cost
    #      term are correct.
    #
    #  (b) Add in the weight decay term (in both the cost function and the derivative
    #      calculations), then re-run Gradient Checking to verify correctness.
    #
    #  (c) Add in the sparsity penalty term, then re-run Gradient Checking to
    #      verify correctness.
    #
    #  Feel free to change the training settings when debugging your
    #  code.  (For example, reducing the training set size or
    #  number of hidden units may make your code run faster; and setting beta
    #  and/or lambda to zero may be helpful for debugging.)  However, in your
    #  final submission of the visualized weights, please use parameters we
    #  gave in Step 0 above.

    (cost, grad) = sparse_autoencoder_cost_fn(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, patches)

    print cost, grad
    # ======================================================================
    # STEP 3: Gradient Checking
    #
    # Hint: If you are debugging your code, performing gradient checking on smaller models
    # and smaller training sets (e.g., using only 10 training examples and 1-2 hidden
    # units) may speed things up.

    # First, lets make sure your numerical gradient computation is correct for a
    # simple function.  After you have implemented computeNumericalGradient.m,
    # run the following:


    if debug:
        gradient.check_gradient()

        # Now we can use it to check your cost function and derivative calculations
        # for the sparse autoencoder.
        # J is the cost function

        J = lambda x: sparse_autoencoder_cost_fn(x, visible_size, hidden_size,
                                                 lambda_, sparsity_param,
                                                 beta, patches)
        num_grad = gradient.compute_gradient(J, theta)

        # Use this to visually compare the gradients side by side
        print num_grad, grad

        # Compare numerically computed gradients with the ones obtained from backpropagation
        diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
        print diff
        print "Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n\n"

    # ======================================================================
    # STEP 4: After verifying that your implementation of
    #  sparseAutoencoderCost is correct, You can start training your sparse
    #  autoencoder with minFunc (L-BFGS).

    #  Randomly initialize the parameters
    theta = sparse_autoencoder.initialize(hidden_size, visible_size)

    J = lambda x: sparse_autoencoder_cost_fn(x, visible_size, hidden_size,
                                             lambda_, sparsity_param,
                                             beta, patches)
    options_ = {'maxiter': 400, 'disp': True}
    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    opt_theta = result.x

    print result

    # ======================================================================
    # STEP 5: Visualization

    W1 = opt_theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size).transpose()
    display_network.display_network(W1)

