"""
Tutorial Logistic Regression with Stochastic Gradient Descent

Author: Sophie Marchand
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as skd


# Function initialization parameters, update parameters and compute error
def init_parameters():
    return 0, 0, 0


def randomized_training_data(x_1, x_2, y):
    index_array = np.arange(len(x_1))
    np.random.shuffle(index_array)
    return x_1[index_array], x_2[index_array], y[index_array]


def update_parameters_stochastic(a, b_1, b_2, x_1, x_2, y, learning_rate, iteration):
    x_1_it = x_1[iteration]
    x_2_it = x_2[iteration]
    y_it = y[iteration]
    f = -(a + b_1 * x_1_it + b_2 * x_2_it)
    p = np.exp(-f) / (1 + np.exp(-f))
    a_update = a - learning_rate * (p - y_it) * p * (1 - p)
    b_1_update = b_1 - learning_rate * (p - y_it) * p * (1 - p) * x_1_it
    b_2_update = b_2 - learning_rate * (p - y_it) * p * (1 - p) * x_2_it
    return a_update, b_1_update, b_2_update


def compute_accuracy(a, b_1, b_2, x_1, x_2, y, p_th):
    f = -(a + b_1 * x_1 + b_2 * x_2)
    p_pred = np.exp(-f) / (1 + np.exp(-f))
    y_pred = (p_pred > p_th).astype(int)
    accuracy_pred = sum(y_pred == y) / len(y)
    return accuracy_pred


# Create data for binary classification with two input variables
X, y = skd.make_blobs(n_samples=40, n_features=2, centers=2,
                      cluster_std=1.2, random_state=3)
x_1 = np.asarray([X[i, 0] for i in range(0, len(X))])
x_2 = np.asarray([X[i, 1] for i in range(0, len(X))])
p_th = 0.5
learning_rate = 0.3
number_batch = 1
accuracy = []

# Workflow stochastic gradient descent
a, b_1, b_2 = init_parameters()
for batch in range(number_batch):
    x_1_rand, x_2_rand, y_rand = randomized_training_data(x_1, x_2, y)
    for iteration in range(len(x_1_rand)):
        a_update, b_1_update, b_2_update = \
            update_parameters_stochastic(a, b_1, b_2, x_1_rand, x_2_rand, y_rand, learning_rate, iteration)
        a, b_1, b_2 = a_update, b_1_update, b_2_update
        accuracy.append(compute_accuracy(a, b_1, b_2, x_1, x_2, y, p_th))
a_final, b_1_final, b_2_final = a, b_1, b_2

# Print results
f = -(a_final + b_1_final * x_1 + b_2_final * x_2)
p_pred = np.exp(-f) / (1 + np.exp(-f))
y_pred = (p_pred > p_th).astype(int)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Stochastic gradient descent for logistic regression applied to binary classification with '
             + str(number_batch) + ' pass and a learning rate of ' + str(learning_rate), fontsize=12)
label_legend = [0, 1]
for color in ['g', 'y']:
    if color == 'g':
        index = np.where(y_pred == 1)
        label = label_legend[1]
    else:
        index = np.where(y_pred == 0)
        label = label_legend[0]
    ax1.scatter(x_1[index], x_2[index], c=color, label=label)
ax1.set(xlabel='$x_{1}$', ylabel='$x_{2}$', title='Logistic regression results')
ax1.legend()
ax2.plot(range(1, len(accuracy) + 1), accuracy, 'ro-')
ax2.set(xlabel='Number iteration over input variables', ylabel='Accuracy',
        title='Accuracy stochastic gradient descent')
plt.show()
