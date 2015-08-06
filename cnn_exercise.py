import cnn
import numpy as np
import load_MNIST


image_dim = 28

filter_dim = 8
num_filters = 100

num_images = 60000

pool_dim = 3

images = load_MNIST.load_MNIST_images('./data/mnist/train-images-idx3-ubyte')
images = images.reshape(image_dim, image_dim, num_images)

W = np.random.randn(filter_dim, filter_dim, num_filters)
b = np.random.randn(num_filters)

##======================================================================
## STEP 1: Implement and test convolution
#  In this step, you will implement the convolution and test it on
#  on a small part of the data set to ensure that you have implemented
#  this step correctly.

## STEP 1a: Implement convolution
#  Implement convolution in the function cnn_convolve in cnn.py

## Use only the first 8 images for testing
conv_images = images[:, :, 0:8]

# NOTE: Implement cnn_convolve in cnn.py first!
convolved_features = cnn.cnn_convolve(filter_dim, num_filters, conv_images, W, b)

## STEP 1b: Checking your convolution
#  To ensure that you have convolved the features correctly, we have
#  provided some code to compare the results of your convolution with
#  activations from the sparse autoencoder
# (SR-71) wtf is this business about the sparse autoencoder?  holdover from earlier iteration?

for _ in range(1000):
    filter_num = np.random.randint(0, num_filters)
    image_num = np.random.randint(0, 8)
    image_row = np.random.randint(0, image_dim - filter_dim + 1)
    image_col = np.random.randint(0, image_dim - filter_dim + 1)

    patch = conv_images[image_row:image_row + filter_dim, image_col:image_col + filter_dim, image_num]

    feature = np.sum(np.sum(patch * W[:, :, filter_num])) + b[filter_num]
    feature = 1 / (1 + np.exp(-feature))

    convolved_feature = convolved_features[image_row, image_col, filter_num, image_num]
    if np.abs(feature - convolved_feature) > 1e-9:
        print 'Convolved feature does not match test feature'
        print 'Filter Number: %s' % filter_num
        print 'Image Number: %s' % image_num
        print 'Image Row: %s' % image_row
        print 'Image Column: %s' % image_col
        print 'Convolved feature: %0.5f' % convolved_feature
        print 'Test feature: %0.5f' % feature
        exit("Convolved feature does not match test feature")

##======================================================================
## STEP 2: Implement and test pooling
#  Implement pooling in the function cnn_pool in cnn.py

## STEP 2a: Implement pooling
# NOTE: Implement cnn_pool in cnn.py first!
# see https://github.com/jatinshah/ufldl_tutorial/blob/master/cnn_exercise.py

test_matrix = np.array(range(0, 64)).reshape(8, 8)

expected_matrix = np.array([[np.mean(test_matrix[0:4, 0:4]), np.mean(test_matrix[0:4, 4:8])],
                            [np.mean(test_matrix[4:8, 0:4]), np.mean(test_matrix[4:8, 4:8])]])

test_matrix = np.reshape(test_matrix, (8, 8, 1, 1))

pooled_features = cnn.cnn_pool(4, test_matrix).squeeze()

## STEP 2b: Checking your pooling
#  To ensure that you have implemented pooling, we will use your pooling
#  function to pool over a test matrix and check the results.
if not (pooled_features == expected_matrix).all():
    print "Pooling incorrect"
    print "Expected matrix"
    print expected_matrix
    print "Got"
    print pooled_features
    exit("Pooling code failed")

print 'Congratulations! Your code passed the test.'
