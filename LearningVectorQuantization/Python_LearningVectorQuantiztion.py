"""
Tutorial Learning Vector Quantization
Explanations at: https://sophiemarchand.netlify.app/
Author: Sophie Marchand
"""
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import sklearn.datasets as skd
import matplotlib.pyplot as plt

# Create a two-class dataset 
n = 60
X, y = skd.make_blobs(n_samples=n, n_features=2, centers=2,
                      cluster_std=1.6, random_state=42)

# Extract a balanced train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Select initial codebook vectors
number_codebook = 4
ratio_codebook = number_codebook/len(y_train)
X_train_c, X_codebook, y_train_c, y_codebook = train_test_split(X_train, y_train, test_size=ratio_codebook, random_state=42, stratify=y_train)

# Update codebook vectors for every training instance to get the model
initial_learning_rate = 0.01
number_epoch = 15
for epoch in range(number_epoch):
    learning_rate = initial_learning_rate * (1 - epoch / number_epoch)
    for id_train in range(X_train_c.shape[0]):
        list_distance_euclidean = []
        for id_codebook in range(len(X_codebook)):
            distance_euclidean = sum(np.sqrt(np.square(X_train[id_train]-X_codebook[id_codebook])))
            list_distance_euclidean.append(distance_euclidean)
        best_codebook_match_index = np.where(list_distance_euclidean == min(list_distance_euclidean))[0][0]
        best_codebook_match = X_codebook[best_codebook_match_index]
        if y_codebook[id_codebook] == y_train_c[id_train]:
            for id_attribute in range(len(best_codebook_match)):
                update_attribute = best_codebook_match[id_attribute] + \
                    learning_rate * (X_train_c[id_train][id_attribute] - best_codebook_match[id_attribute])
                X_codebook[best_codebook_match_index][id_attribute] = update_attribute
        else:
            for id_attribute in range(len(best_codebook_match)):
                update_attribute = best_codebook_match[id_attribute] - \
                        learning_rate * (X_train_c[id_train][id_attribute] - best_codebook_match[id_attribute])
                X_codebook[best_codebook_match_index][id_attribute] = update_attribute

# Make prediction on the test dataset
y_prediction = []
for id_test in range(X_test.shape[0]):
    list_distance_euclidean = []
    for id_codebook in range(X_codebook.shape[0]):
        distance_euclidean = sum(np.sqrt(np.square(X_test[id_test]-X_codebook[id_codebook])))
        list_distance_euclidean.append(distance_euclidean)
    best_codebook_match_index = np.where(list_distance_euclidean == min(list_distance_euclidean))[0][0]
    y_match = y_codebook[best_codebook_match_index]
    y_prediction.append(y_match)

# Print results
y_accuracy = sum(y_prediction == y_test) / len(y_test)
y_prediction = np.asarray(y_prediction)
fig, ax = plt.subplots()
fig.suptitle('Learning Vector Quantization: accuracy = ' + str(y_accuracy) + ' with $\\alpha$ = ' + str(initial_learning_rate), fontsize=15)
label_legend = ['class 1 train', 'class 2 train', 'class 1 prediction', 'class 2 prediction']
for color in ['r', 'g']:
    if color == 'r':
        index = np.where(y_train_c == 0)
        label = label_legend[0]
    else:
        index = np.where(y_train_c == 1)
        label = label_legend[1]
    plt.plot(X_train_c[index[0],0], X_train_c[index[0],1], '.', \
        markersize=15, c=color, label=label)
for color in ['r', 'g']:
    if color == 'r':
        index = np.where(y_prediction == 0)
        label = label_legend[2]
    else:
        index = np.where(y_prediction == 1)
        label = label_legend[3]
    plt.plot(X_test[index[0],0], X_test[index[0],1], '*', \
        markersize=15, c=color, label=label, markeredgecolor= 'k')
ax.set(xlabel='attribute 1', ylabel='attribute 2')
ax.legend(loc='lower left')
plt.show()
