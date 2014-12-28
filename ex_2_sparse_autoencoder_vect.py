import load_MNIST
import train

if __name__ == '__main__':
    # ======================================================================
    # STEP 0: Here we provide the relevant parameters values that will
    #  allow your sparse autoencoder to get good filters; you do not need to
    #  change the parameters below.
    visible_size = 28 * 28
    hidden_size = 196
    sparsity_param = 0.1
    lambda_ = 3e-3
    beta = 3
    debug = False

    # ======================================================================
    # STEP 1: Load MNIST data
    #
    images = load_MNIST.load_MNIST_images('data/mnist/train-images-idx3-ubyte')
    patches = images[:, 0:10000]
    use_vectorized_implementation = True
    train.train(visible_size, hidden_size, sparsity_param, lambda_, beta, debug,
                patches, use_vectorized_implementation)

