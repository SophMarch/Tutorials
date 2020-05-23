"""
Tutorial AdaBoost for Decision Trees
Explanations at: https://sophiemarchand.netlify.app/
Author: Sophie Marchand
"""
import numpy as np
import sklearn.datasets as skd
from matplotlib import pyplot as plt

# Functions to create AdaBoost workflow
def get_stump_prediction(dataset, col_split, variable_split):
    col_split_copy = np.asarray(dataset[:,col_split])
    y_prediction = []
    for row in range(dataset.shape[0]):
        if col_split_copy[row] < variable_split:
            y_prediction.append(0)
        else:
            y_prediction.append(1)
    y_prediction = np.asarray(y_prediction)
    return y_prediction
        
def evaluate_prediction(y, y_prediction, X_weight):
    error_index = np.where(y != y_prediction)[0]
    error = np.asarray([0] * len(y))
    error[error_index] = 1
    missclassification_rate = sum(X_weight * error)/sum(X_weight)
    stage = np.log((1-missclassification_rate)/missclassification_rate)
    return error, stage

def update_weight(X_weight, error, stage):
    return X_weight * np.exp(error*stage)

def get_prediction(X, split, stage):
    y_prediction = get_stump_prediction(X, split[1], split[0])
    index_y_change = np.where(y_prediction == 0)[0]
    y_prediction[index_y_change] = -1
    y_prediction = y_prediction * stage
    return y_prediction


# Create dataset for binary classification
n = 40
X, y = skd.make_blobs(n_samples=n, n_features=2, centers=2, cluster_std=1, random_state=65)
X_weight = [1/n] * len(X)

# Apply stump
model_split = [[X[1][0], 0], [X[15][1], 1], [X[14][0], 0]]
stump_stage = []
for split in model_split:
    y_prediction = get_stump_prediction(X, split[1], split[0])
    error, stage = evaluate_prediction(y, y_prediction, X_weight)
    X_weight = update_weight(X_weight, error, stage)
    stump_stage.append(stage)

# Make prediction on new data
n = 10
X_test, y_test = skd.make_blobs(n_samples=n, n_features=2, centers=2, cluster_std=1, random_state=65)
inter_prediction = [0] * n
for i in range(len(model_split)):
    prediction_model = get_prediction(X_test, model_split[i], stump_stage[i])
    inter_prediction += prediction_model
index_0 = np.where(inter_prediction < 0)[0]
y_prediction = np.asarray([1] * n)
y_prediction[index_0] = 0 

# Print results
y_accuracy = sum(y_test == y_prediction) / n
y = np.asarray(y)
fig, ax = plt.subplots()
fig.suptitle('AdaBoost for Decision Trees: accuracy = ' + str(y_accuracy), fontsize=15)
label_legend = ['0', '1', '0 prediction', '1 prediction']
for color in ['b', 'g']:
    if color == 'b':
        index = np.where(y == 0)
        label = label_legend[0]
    else:
        index = np.where(y == 1)
        label = label_legend[1]
    plt.plot(X[index[0], 0], X[index[0], 1], '.', \
        markersize=15, c=color, label=label)
for color in ['b', 'g']:
    if color == 'b':
        index = np.where(y_prediction == 0)
        label = label_legend[2]
    else:
        index = np.where(y_prediction == 1)
        label = label_legend[3]
    plt.plot(X_test[index[0],0], X_test[index[0],1], '*', \
        markersize=15, c=color, label=label, markeredgecolor= 'k')
ax.set(xlabel='$x_{1}$', ylabel='$x_{2}$')
ax.legend(loc='lower right')
plt.show()
