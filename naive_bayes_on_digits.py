#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:57:46 2024

@author: soumensmacbookair
"""

#%% Imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import tensorflow as tf

#%% Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train >= 128).astype(int)
x_test = (x_test >= 128).astype(int)

x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

#%% Naive Bayes classifier
nb_classifier = BernoulliNB()
nb_classifier.fit(x_train, y_train)
y_pred = nb_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes: Accuracy: {accuracy:.2f}")
