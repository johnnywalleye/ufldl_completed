import display_network
import matplotlib.pyplot as plt
import numpy as np
import random
import sample_images

# ================================================================
# Step 0a: Load data
#  Here we provide the code to load natural image data into x.
#  x will be a 144 * 10000 matrix, where the kth column x(:, k) corresponds to
#  the raw image data from the kth 12x12 image patch sampled.
#  You do not need to change the code below.
patches = sample_images.sample_images_raw()
num_samples = patches.shape[1]
random_sel = random.sample(range(num_samples), 400)
display_network.display_network(patches[:, random_sel], 'raw_pca.png')

# ================================================================
# Step 0b: Zero-mean the data (by row)

# -------------------- YOUR CODE HERE --------------------
patches_0 = patches - patches.mean(axis=0)

# ================================================================
# Step 1a: Implement PCA to obtain xRot
#  Implement PCA to obtain xRot, the matrix in which the data is expressed
#  with respect to the eigenbasis of sigma, which is the matrix U.

# -------------------- YOUR CODE HERE -------------------- 
# xRot = zeros(size(x)); # You need to compute this
num_data_points = len(patches[0])
sigma = np.einsum('ik,jk->ij', patches, patches) / num_data_points
eigenvectors, eigenvalues, _ = np.linalg.svd(sigma)
u = eigenvectors
x_rot = np.dot(u.T, patches)

# ================================================================
#  Step 1b: Check your implementation of PCA
#  The covariance matrix for the data expressed with respect to the basis U
#  should be a diagonal matrix with non-zero entries only along the main
#  diagonal. We will verify this here.
#  Write code to compute the covariance matrix, covar. 
#  When visualised as an image, you should see a straight line across the
#  diagonal (non-zero entries) against a blue background (zero entries).

# -------------------- YOUR CODE HERE -------------------- 
# covar = zeros(size(x, 1)); # You need to compute this
covar = np.einsum('ik,jk->ij', x_rot, x_rot) / num_data_points
import pdb; pdb.set_trace()
fig1 = plt.figure(1)
plt.imshow(covar)
fig1.suptitle('Visualization of Covariance Matrix')
fig1.show()
# imagesc(covar);

# ================================================================
#  Step 2: Find k, the number of components to retain
#  Write code to determine k, the number of components to retain in order
#  to retain at least 99# of the variance.

# -------------------- YOUR CODE HERE -------------------- 
# k = 0; # Set k accordingly
var_to_capture = 0.99
eig_cumsum = eigenvalues.cumsum() / eigenvalues.sum()
num_to_retain = np.argmax(eig_cumsum[eig_cumsum < var_to_capture]) + 1

# ================================================================
#  Step 3: Implement PCA with dimension reduction
#  Now that you have found k, you can reduce the dimension of the data by
#  discarding the remaining dimensions. In this way, you can represent the
#  data in k dimensions instead of the original 144, which will save you
#  computational time when running learning algorithms on the reduced
#  representation.
# 
#  Following the dimension reduction, invert the PCA transformation to produce 
#  the matrix xHat, the dimension-reduced data with respect to the original basis.
#  Visualise the data and compare it to the raw data. You will observe that
#  there is little loss due to throwing away the principal components that
#  correspond to dimensions with low variation.

# -------------------- YOUR CODE HERE -------------------- 
# xHat = zeros(size(x));  # You need to compute this
x_tilde = x_rot.copy()
x_tilde[num_to_retain:, ] = 0.0
x_hat = np.dot(u, x_tilde)
display_network.display_network(patches[:, random_sel], 'Raw Images.png')
display_network.display_network(x_hat[:, random_sel], 'PCA processed images %.2f .png' % var_to_capture)

# Visualise the data, and compare it to the raw data
# You should observe that the raw and processed data are of comparable quality.
# For comparison, you may wish to generate a PCA reduced image which
# retains only 90# of the variance.

# figure('name',['PCA processed images ',sprintf('(#d / #d dimensions)', k, size(x, 1)),'']);
# display_network(xHat(:,randsel));
# figure('name','Raw images');
# display_network(x(:,randsel));

##================================================================
## Step 4a: Implement PCA with whitening and regularisation
#  Implement PCA with whitening and regularisation to produce the matrix
#  xPCAWhite.
epsilon = 1e-15
x_pca_white = x_rot / np.atleast_2d(np.sqrt(eigenvalues + epsilon)).T
covar_white = np.einsum('ik,jk->ij', x_pca_white, x_pca_white) / num_data_points
fig2 = plt.figure(2)
plt.imshow(covar_white)
fig2.suptitle('Visualization of PCA whitened Cov Matrix (without regularization)')
fig2.show()

epsilon = 0.1
x_pca_white = x_rot / np.atleast_2d(np.sqrt(eigenvalues + epsilon)).T
covar_white = np.einsum('ik,jk->ij', x_pca_white, x_pca_white) / num_data_points
fig3 = plt.figure(3)
plt.imshow(covar_white)
fig3.suptitle('Visualization of PCA whitened Cov Matrix (with regularization)')
fig3.show()

# epsilon = 0.1;
# xPCAWhite = zeros(size(x));

# -------------------- YOUR CODE HERE -------------------- 

# ================================================================
#  Step 4b: Check your implementation of PCA whitening
#  Check your implementation of PCA whitening with and without regularisation. 
#  PCA whitening without regularisation results a covariance matrix 
#  that is equal to the identity matrix. PCA whitening with regularisation
#  results in a covariance matrix with diagonal entries starting close to 
#  1 and gradually becoming smaller. We will verify these properties here.
#  Write code to compute the covariance matrix, covar. 
#
#  Without regularisation (set epsilon to 0 or close to 0), 
#  when visualised as an image, you should see a red line across the
#  diagonal (one entries) against a blue background (zero entries).
#  With regularisation, you should see a red line that slowly turns
#  blue across the diagonal, corresponding to the one entries slowly
#  becoming smaller.

# -------------------- YOUR CODE HERE -------------------- 

# Visualise the covariance matrix. You should see a red line across the
# diagonal against a blue background.
# figure('name','Visualisation of covariance matrix');
# imagesc(covar);

# ================================================================
#  Step 5: Implement ZCA whitening
#  Now implement ZCA whitening to produce the matrix xZCAWhite. 
#  Visualise the data and compare it to the raw data. You should observe
#  that whitening results in, among other things, enhanced edges.

x_zca_white = np.dot(u, x_pca_white)
display_network.display_network(x_zca_white[:, random_sel], 'ZCA whitened images')
display_network.display_network(patches[:, random_sel], 'Raw images')


# xZCAWhite = zeros(size(x));

# -------------------- YOUR CODE HERE -------------------- 

# Visualise the data, and compare it to the raw data.
# You should observe that the whitened images have enhanced edges.
# figure('name','ZCA whitened images');
# display_network(xZCAWhite(:,randsel));
# figure('name','Raw images');
# display_network(x(:,randsel));
