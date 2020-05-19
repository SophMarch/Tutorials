"""
Tutorial K-Nearest Neighbors
Explanations at: https://sophiemarchand.netlify.app/
Author: Sophie Marchand
"""
import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from statistics import mode
import matplotlib.pyplot as plt

# Import the iris dataset 
iris = datasets.load_iris()

# Extract a balanced train and test dataset
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42, stratify=iris.target)

# Predict for each test variable input the associated class with the Euclidean distance 
y_prediction = []
k = 4
for id_test in range(X_test.shape[0]):
    list_distance_euclidean = []
    for id_train in range(X_train.shape[0]):
        distance_euclidean = sum(np.sqrt(np.square(X_train[id_train]-X_test[id_test])))
        list_distance_euclidean.append(distance_euclidean)
    k_index = sorted(range(len(list_distance_euclidean)), key = lambda i: list_distance_euclidean[i])[:k]
    k_y_value = mode(y_train[k_index])
    y_prediction.append(k_y_value)

# Print results
y_accuracy = sum(y_prediction == y_test) / len(y_test)
y_prediction = np.asarray(y_prediction)
fig, ax = plt.subplots()
fig.suptitle('K-Nearest Neighbors: accuracy = ' + str(y_accuracy) + ' with k = ' + str(k), fontsize=15)
label_legend = iris.target_names
for color in ['r', 'y', 'g']:
    if color == 'r':
        index = np.where(y_train == 0)
        label = label_legend[0] + ' train'
    elif color == 'y':
        index = np.where(y_train == 1)
        label = label_legend[1] + ' train'
    else:
        index = np.where(y_train == 2)
        label = label_legend[2] + ' train'
    plt.plot(X_train[index[0],0], X_train[index[0],3], '.', \
        markersize=15, c=color, label=label)
for color in ['r', 'y', 'g']:
    if color == 'r':
        index = np.where(y_prediction == 0)
        label = label_legend[0] + ' prediction'
    elif color == 'y':
        index = np.where(y_prediction == 1)
        label = label_legend[1] + ' prediction'
    else:
        index = np.where(y_prediction == 2)
        label = label_legend[2] + ' prediction'
    plt.plot(X_test[index[0],0], X_test[index[0],3], '*', \
        markersize=15, c=color, label=label, markeredgecolor= 'k')
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[2])
ax.legend(loc='lower right')
plt.show()
