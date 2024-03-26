#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:57:46 2024

@author: soumensmacbookair
"""

#%% Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

#%% Get the digits dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def preprocess_data(
    x: np.array,
    y: np.array
) -> Tuple[np.array, np.array]:

    x = x.astype(np.float64)
    y = y.astype(np.int64)
    x = x/255.0
    x = x.reshape((-1,784))
    return (x, y)

(x_train, y_train) = preprocess_data(*(x_train, y_train))
(x_test, y_test) = preprocess_data(*(x_test, y_test))

def reduce_dimension(
    vector: np.array,
    dimension: int
) -> Tuple[np.array, np.array, np.array]:

    # Get the data covariance matrix
    S = np.cov(x_train, rowvar=False) # x_train.shape = (N,d), S.shape = (d,d)

    # Perform Eigenvalue analysis on data covariance matrix
    l, w = np.linalg.eig(S) # l.shape = (d,), w.shape = (d,d)

    index = np.argsort(l)[::-1]
    L = l[index]
    W = w[:, index]

    # Take top k eigenvalues and eigenvectors
    k = dimension
    L = L[:k] # L.shape = (k,)
    W = W[:,:k] # W.shape = (d,k)

    # Project the data in lower dimension
    proj_vector = np.matmul(vector, W) # vector.shape = (M,d), proj_vector.shape = (M,k)

    return proj_vector, W, L

def calc_recon_error(
    vector: np.array,
    dimension: int
) -> float:

    # Project the data in lower dimension
    proj_vector, W, _ = reduce_dimension(vector, dimension)

    # Reproject the data back to higher dimension
    recon_vector = np.matmul(proj_vector, W.T) # recon_vector.shape = (M,d)

    # Measure the reconstruction error
    recon_error = np.linalg.norm(vector - recon_vector, axis=1).mean()

    return recon_error

k_array = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
error_array = np.zeros(shape=(len(k_array),))

for i,k in enumerate(k_array):
    error_array[i] = calc_recon_error(x_test, k)

# Plot dimension vs error change
plt.plot(k_array, error_array, marker='o')
plt.title('Dimension vs Error Change')
plt.xlabel('Dimension (k)')
plt.ylabel('Reconstruction Error')
plt.xscale('log')
plt.grid(which='both')

for i, (x, y) in enumerate(zip(k_array, error_array)):
    plt.annotate(f'({x},{y:.2f})', xy=(x, y), xytext=(5, 5), textcoords='offset points')

plt.show()

#%% Original vs Reconstructed Image
image = x_test[0,:].reshape((1,-1))
label = y_test[0]
num_pcs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 784])

# Plot original image
plt.figure()
plt.subplot(3, 4, 1)
plt.imshow(image.reshape((28, 28)), cmap='gray')
plt.title(f'Original, Label: {int(label)}')

# Loop through different numbers of principal components
for i, num_pc in enumerate(num_pcs):
    proj_image, W, _ = reduce_dimension(image, dimension=num_pc)
    recon_image = np.matmul(proj_image, W.T).reshape((28, 28))

    # Plot reconstructed images
    plt.subplot(3, 4, i + 2)
    plt.imshow(recon_image.reshape((28, 28)).astype(np.float64), cmap='gray')
    plt.title(f'Recon, # of PC: {num_pc}')

plt.tight_layout()
plt.show()



print("END")
