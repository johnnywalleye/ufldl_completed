#  CS294A/CS294W Softmax Exercise

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  softmax exercise. You will need to write the softmax cost function 
#  in softmaxCost.m and the softmax prediction function in softmaxPred.m. 
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#  (However, you may be required to do so in later exercises)

import gradient
import load_MNIST
import numpy as np
import softmax

# ======================================================================
#  STEP 0: Initialise constants and parameters
#
#  Here we define and initialise some constants which allow your code
#  to be used more generally on any arbitrary input. 
#  We also initialise some parameters used for tuning the model.

input_size = 28 * 28  # Size of input vector (MNIST images are 28x28)
num_classes = 10  # Number of classes (MNIST images fall into 10 classes)

lambda_ = 1e-4  # Weight decay parameter

# ======================================================================
#  STEP 1: Load data
#
#  In this section, we load the input and output data.
#  For softmax regression on MNIST pixels, 
#  the input data is the images, and 
#  the output data is the labels.
#

# Change the filenames if you've saved the files under different names
# On some platforms, the files might be saved as 
# train-images.idx3-ubyte / train-labels.idx1-ubyte

# (REMOVE) images is (784, 60000), i.e. input_size * num_points
images = load_MNIST.load_MNIST_images('data/mnist/train-images-idx3-ubyte')
# (REMOVE) labels is (60000, ), i.e. num_points
labels = load_MNIST.load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')
labels[labels == 0] = 10  # Remap 0 to 10

input_data = images

# For debugging purposes, you may wish to reduce the size of the input data
# in order to speed up gradient checking. 
# Here, we create synthetic dataset using random data for testing

debug = False  # Set debug = True when debugging.
if debug:
    input_size = 8
    input_data = np.random.normal(size=[8, 100])  # 100 dummy images with 8 pixels each
    labels = np.random.randint(1, num_classes + 1, 100)  # 100 labels

# Randomly initialise theta
theta = 0.005 * np.random.normal(size=[num_classes * input_size, 1])

# ======================================================================
#  STEP 2: Implement softmax_cost
#
#  Implement softmax_cost in softmax.py

cost, grad = softmax.softmax_cost(theta, num_classes, input_size, lambda_, input_data, labels)

# ======================================================================
#  STEP 3: Gradient checking
#
#  As with any learning algorithm, you should always check that your
#  gradients are correct before learning the parameters.
# 

if debug:
    J = lambda x: softmax.softmax_cost(x, num_classes, input_size, lambda_, input_data, labels)
    num_grad = gradient.compute_gradient(J, theta)

    # Use this to visually compare the gradients side by side
    # print num_grad, grad

    # Compare numerically computed gradients with those computed analytically
    grad_shaped = grad.reshape(num_grad.shape)
    diff = np.linalg.norm(num_grad - grad_shaped) / np.linalg.norm(num_grad + grad_shaped)
    print "Norm of the difference between numerical and analytical num_grad (should be < 1e-9) is: {0:f}\n".format(diff)
    # The difference should be small.
    # In our implementation, these values are usually less than 1e-7.

    # When your gradients are correct, congratulations!

# ======================================================================
#  STEP 4: Learning parameters
#
#  Once you have verified that your gradients are correct, 
#  you can start training your softmax regression code using softmax_train
#  (which uses minFunc).

options = {'maxiter': 100}
softmax_model = softmax.softmax_train(input_size, num_classes, lambda_, input_data, labels, options)
                          
# Although we only use 100 iterations here to train a classifier for the 
# MNIST data set, in practice, training for more iterations is usually
# beneficial.

# ======================================================================
#  STEP 5: Testing
#
#  You should now test your model against the test images.
#  To do this, you will first need to write softmax_predict
#  (in softmax.py), which should return predictions
#  given a softmax model and the input data.

images = load_MNIST.load_MNIST_images('data/mnist/t10k-images-idx3-ubyte')
labels = load_MNIST.load_MNIST_labels('data/mnist/t10k-labels-idx1-ubyte')
labels[labels == 0] = 10  # Remap 0 to 10

input_data = images

# You will have to implement softmax_predict in softmax.py
pred = softmax.softmax_predict(softmax_model, input_data)

accuracy = np.mean(labels == pred)
print('Accuracy: {0:0.3f}\n'.format(accuracy * 100))

# Accuracy is the proportion of correctly classified images
# After 100 iterations, the results for our implementation were:
#
# Accuracy: 92.200#
#
# If your values are too low (accuracy less than 0.91), you should check 
# your code for errors, and make sure you are training on the 
# entire data set of 60000 28x28 training images 
# (unless you modified the loading code, this should be the case)
