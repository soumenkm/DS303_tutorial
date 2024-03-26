#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:57:46 2024

@author: soumensmacbookair
"""

#%% Imports
import torch
import torchvision
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
from PIL import Image
import seaborn as sns

#%% Common utility
flag = 2
BATCH_SIZE = 32

#%% Standard dataset in Tensorflow
# https://www.tensorflow.org/datasets/catalog/overview

def transform_data(
    data_x: tf.constant,
    data_y: tf.constant,
    is_raw: bool
    ) -> Tuple[tf.constant, tf.constant]:

    x = data_x
    x = tf.reshape(x, shape=(BATCH_SIZE, -1))
    x = tf.cast(x, dtype=tf.float64)

    y = data_y
    y = tf.cast(y, dtype=tf.int64)

    if is_raw:
        return (x, y)
    else:
        x = x/255.0
        mean = tf.math.reduce_mean(x, axis=0, keepdims=True)
        std = tf.math.reduce_std(x, axis=0, keepdims=True)
        x = (x-mean)/(std+tf.keras.backend.epsilon())
        return (x, y)

def get_tf_dataset(
    is_train: bool,
    return_ds: bool,
    is_raw: bool
    ) -> Union[tf.data.Dataset, Tuple[np.array, np.array]]:

    if is_train:
        split = "train"
    else:
        split = "test"

    dataset, info = tfds.load(
        name="mnist",
        split=split,
        as_supervised=True,
        download=True,
        with_info=True
    )
    dataset = dataset.shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.map(map_func=lambda x, y: transform_data(data_x=x, data_y=y, is_raw=is_raw))

    feature_dim = dataset.__iter__().__next__()[0].shape[1]
    x = np.zeros((dataset.__len__()*BATCH_SIZE, feature_dim))
    y = np.zeros((dataset.__len__()*BATCH_SIZE,))

    for i, data in enumerate(dataset):
        x[BATCH_SIZE*i:BATCH_SIZE*(i+1),:] = data[0].numpy()
        y[BATCH_SIZE*i:BATCH_SIZE*(i+1)] = data[1].numpy()

    if return_ds:
        return dataset
    else:
        return (x,y)

if flag==1:
    train_dataset = get_tf_dataset(is_train=True, return_ds=True, is_raw=False)
    test_dataset = get_tf_dataset(is_train=False, return_ds=True, is_raw=False)

    train_x, train_y = get_tf_dataset(is_train=True, return_ds=False, is_raw=False)
    test_x, test_y = get_tf_dataset(is_train=False, return_ds=False, is_raw=False)

    raw_train_dataset = get_tf_dataset(is_train=True, return_ds=True, is_raw=True)
    raw_test_dataset = get_tf_dataset(is_train=False, return_ds=True, is_raw=True)

    raw_train_x, raw_train_y = get_tf_dataset(is_train=True, return_ds=False, is_raw=True)
    raw_test_x, raw_test_y = get_tf_dataset(is_train=False, return_ds=False, is_raw=True)

#%% Standard dataset in PyTorch
# https://pytorch.org/vision/main/datasets.html

def transform_features(
    data_x: Image.Image
    ) -> torch.tensor:

    x = torchvision.transforms.PILToTensor()(data_x)
    x = torch.reshape(x, shape=(-1,))
    x = x.to(dtype=torch.float64)

    return x

def transform_labels(
    data_y: Image.Image
    ) -> torch.tensor:

    y = torch.tensor(data_y)
    y = y.to(dtype=torch.int64)

    return y

def batch_normalization(
    batch: List[Tuple[torch.tensor, torch.tensor]],
    is_raw: bool
    ) -> Tuple[torch.tensor, torch.tensor]:

    batch_x_tensor, batch_y_tensor = torch.utils.data.default_collate(batch)

    if is_raw:
        return [batch_x_tensor, batch_y_tensor]
    else:
        batch_x_tensor = batch_x_tensor/255.0
        mean = batch_x_tensor.mean(dim=0, keepdim=True)
        std = batch_x_tensor.std(dim=0, keepdim=True)
        batch_x_tensor = (batch_x_tensor-mean)/(std+torch.finfo(torch.float64).eps)

        return [batch_x_tensor, batch_y_tensor]

def get_torch_dataloader(
    is_train: bool,
    return_dl: bool,
    is_raw: bool
    ) -> Union[torch.utils.data.DataLoader, Tuple[np.array, np.array]]:

    dataset = torchvision.datasets.MNIST(
        root="mnist",
        train=is_train,
        download=True,
        transform=torchvision.transforms.Lambda(transform_features),
        target_transform=torchvision.transforms.Lambda(transform_labels)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: batch_normalization(batch=batch, is_raw=is_raw)
    )

    feature_dim = dataloader.__iter__().__next__()[0].shape[1]
    x = np.zeros((dataloader.__len__()*BATCH_SIZE, feature_dim))
    y = np.zeros((dataloader.__len__()*BATCH_SIZE,))

    for i, data in enumerate(dataloader):
        x[BATCH_SIZE*i:BATCH_SIZE*(i+1),:] = data[0].numpy()
        y[BATCH_SIZE*i:BATCH_SIZE*(i+1)] = data[1].numpy()

    if return_dl:
        return dataloader
    else:
        return (x,y)

if flag == 2:
    train_dataloader = get_torch_dataloader(is_train=True, return_dl=True, is_raw=False)
    test_dataloader = get_torch_dataloader(is_train=False, return_dl=True, is_raw=False)

    train_x, train_y = get_torch_dataloader(is_train=True, return_dl=False, is_raw=False)
    test_x, test_y = get_torch_dataloader(is_train=False, return_dl=False, is_raw=False)

    raw_train_dataloader = get_torch_dataloader(is_train=True, return_dl=True, is_raw=True)
    raw_test_dataloader = get_torch_dataloader(is_train=False, return_dl=True, is_raw=True)

    raw_train_x, raw_train_y = get_torch_dataloader(is_train=True, return_dl=False, is_raw=True)
    raw_test_x, raw_test_y = get_torch_dataloader(is_train=False, return_dl=False, is_raw=True)

#%% Data visualization
# Display sample images
def display_sample_images(
    X: np.array,
    y: np.array
) -> None:

    fig, axes = plt.subplots(3, 5)
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"Label: {y[i]}")
        ax.axis('off')
    plt.show()

 # Distribution of labels
def label_distribution(
    y: np.array
) -> None:

    plt.figure()
    plt.hist(y, bins=np.arange(11)-0.5, rwidth=0.8, color='skyblue', edgecolor='black', density=True)
    plt.xticks(np.arange(10))
    plt.xlabel('Digit')
    plt.ylabel('Density')
    plt.title('Distribution of Labels')
    plt.show()

# Features and correlation plot
def visualize_features_and_correlation(
    X: np.array
) -> None:

    # Select 5 random features where the average value is above a certain threshold
    threshold = 50
    average_values = np.mean(X, axis=0)
    above_threshold_indices = np.where(average_values > threshold)[0]
    selected_features_indices = np.random.choice(above_threshold_indices, size=6, replace=False)
    selected_features = X[:, selected_features_indices]

    # Plot PDFs for selected features
    fig, axes = plt.subplots(2, 3)
    for i, ax in enumerate(axes.flatten()):
        sns.histplot(selected_features[:, i], stat="probability", kde=True, ax=ax)
        ax.set_title(f'F: {selected_features_indices[i]} PDF')

    # Plot correlation matrix for selected features
    selected_features_df = pd.DataFrame(selected_features, columns=[f'F: {i}' for i in selected_features_indices])
    correlation_matrix = selected_features_df.corr(method="pearson")
    correlation_matrix[np.isinf(correlation_matrix)] = np.nan

    plt.figure()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix for Selected Features')
    plt.show()

#%% Display samples
display_sample_images(raw_train_x, raw_train_y)

#%% Display labels
label_distribution(raw_train_y)

#%% Display correlation
visualize_features_and_correlation(raw_train_x)

print("END")
