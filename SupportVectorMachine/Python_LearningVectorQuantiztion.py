"""
Tutorial Support Vector Machine
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
                      cluster_std=1, random_state=20)
y[np.where(y==0)[0]] = -1

# Extract a balanced train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Learn the SVM model 
n_epoch = 10
learning_rate = 0.45
b_1, b_2 = 0, 0
for epoch in range(n_epoch):
    for i_index in range(len(X_train)):
        is_supportvector = y_train[i_index] * (b_1 * X_train[i_index][0] + b_2 * X_train[i_index][1])
        if is_supportvector >= 1:
            b_1 = (1-1/(i_index+1))*b_1
            b_2 = (1-1/(i_index+1))*b_2
        else:
            b_1 = (1-1/(i_index+1))*b_1 + 1/(learning_rate * (i_index+1)) * X_train[i_index][0] * y_train[i_index]
            b_2 = (1-1/(i_index+1))*b_2 + 1/(learning_rate * (i_index+1)) * X_train[i_index][1] * y_train[i_index]

# Make prediction on the test dataset
y_prediction = X_test[:,0] * b_1 + X_test[:,1] * b_2
index_minus1 = [np.where(y_prediction < 0)]
index_plus1 = [np.where(y_prediction >= 0)]
y_prediction[index_minus1] = -1
y_prediction[index_plus1] = 1

# Print results
y_accuracy = sum(y_prediction == y_test) / len(y_test)
y_prediction = np.asarray(y_prediction)
fig, ax = plt.subplots()
fig.suptitle('Support Vector Machine: accuracy = ' + str(y_accuracy) + ' with $\\alpha$ = ' + str(learning_rate), fontsize=15)
label_legend = ['class -1 train', 'class 1 train', 'class -1 prediction', 'class 1 prediction']
for color in ['b', 'y']:
    if color == 'b':
        index = np.where(y_train == -1)
        label = label_legend[0]
    else:
        index = np.where(y_train == 1)
        label = label_legend[1]
    plt.plot(X_train[index[0],0], X_train[index[0],1], '.', \
        markersize=15, c=color, label=label)
for color in ['b', 'y']:
    if color == 'b':
        index = np.where(y_prediction == -1)
        label = label_legend[2]
    else:
        index = np.where(y_prediction == 1)
        label = label_legend[3]
    plt.plot(X_test[index[0],0], X_test[index[0],1], '*', \
        markersize=15, c=color, label=label, markeredgecolor= 'k')
ax.set(xlabel='$x_{1}$', ylabel='$x_{2}$')
ax.legend(loc='upper right')
plt.show()
