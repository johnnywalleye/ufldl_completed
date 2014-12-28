import sample_images
import train

if __name__ == '__main__':
    # ======================================================================
    # STEP 0: Here we provide the relevant parameters values that will
    #  allow your sparse autoencoder to get good filters; you do not need to
    #  change the parameters below.
    visible_size = 64
    hidden_size = 25
    sparsity_param = 0.01
    lambda_ = 1e-3
    beta = 3
    debug = False

    # ======================================================================
    # STEP 1: Implement sampleIMAGES
    #
    #  After implementing sampleIMAGES, the display_network command should
    #  display a random sample of 200 patches from the dataset
    patches = sample_images.sample_images()
    train.train(visible_size, hidden_size, sparsity_param, lambda_, beta, debug, patches)
