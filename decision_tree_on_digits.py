#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:57:46 2024

@author: soumensmacbookair
"""

#%% Imports
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#%% Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

#%% Creating the Decision Tree Classifier
def DTclassifier(criteria, trainX, trainY, testX, testY):

    clf = tree.DecisionTreeClassifier(criterion = criteria)
    clf = clf.fit(trainX, trainY)
    Y_pred = clf.predict(testX)
    accuracy = accuracy_score(testY, Y_pred)
    print(f"Accuracy for {criteria}:", accuracy)

    cm = confusion_matrix(testY, Y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm , display_labels = clf.classes_)
    disp.plot()
    plt.title(f"Confusion Matrix for {criteria}")
    plt.show()

DTclassifier("gini", x_train, y_train, x_test, y_test)
DTclassifier("entropy", x_train, y_train, x_test, y_test)

