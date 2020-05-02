"""
Tutorial Gradient Descent for Linear Regression

Author: Sophie Marchand
"""
import matplotlib.pyplot as plt
import numpy as np


# Function initialization parameters, update parameters and compute error
def init_parameters():
    return 0, 0


def randomized_training_data(x, y):
    index_array = np.arange(len(x))
    np.random.shuffle(index_array)
    return x[index_array], y[index_array]


def update_parameters_batch(a, b, x, y, learning_rate):
    a_update = a - learning_rate * sum((a + b * x) - y) / len(x)
    b_update = b - learning_rate * sum(((a + b * x) - y) * x) / len(x)
    return a_update, b_update


def update_parameters_stochastic(a, b, x, y, learning_rate, iteration):
    x_it = x[iteration]
    y_it = y[iteration]
    a_update = a - learning_rate * ((a + b * x_it) - y_it)
    b_update = b - learning_rate * ((a + b * x_it) - y_it) * x_it
    return a_update, b_update


def compute_error_rmse(a, b, x, y):
    return np.sqrt(sum(np.square((a + b * x) - y)) / len(x))


# Create a set of points (x,y) and set a learning rate
x = np.array([2, 3, 5, 6, 7])
y = np.array([4, 6, 7, 9, 10])
learning_rate = 0.01
number_batch = 10
rmse_batch = []
rmse_stochastic = []

# Workflow batch gradient descent to estimate (a,b) parameters of y = a + b*x
a, b = init_parameters()
for batch in range(number_batch):
    a_update, b_update = update_parameters_batch(a, b, x, y, learning_rate)
    rmse_batch.append(compute_error_rmse(a_update, b_update, x, y))
    a, b = a_update, b_update
a_batch, b_batch = a, b

# Workflow stochastic gradient descent to estimate (a,b) parameters of y = a + b*x
a, b = init_parameters()
for batch in range(number_batch):
    x_rand, y_rand = randomized_training_data(x, y)
    for iteration in range(len(x)):
        a_update, b_update = update_parameters_stochastic(a, b, x_rand, y_rand, learning_rate, iteration)
        a, b = a_update, b_update
    rmse_stochastic.append(compute_error_rmse(a, b, x, y))
a_stochastic, b_stochastic = a, b

# Print results
y_batch = a_batch + b_batch * x
y_stochastic = a_stochastic + b_stochastic * x

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Gradient descent for linear regression with two methods: batch & stochastic for '
             + str(number_batch) + ' passes and a learning rate of ' + str(learning_rate), fontsize=12)
ax1.plot(x, y, 'bo', label='Initial data')
ax1.plot(x, y_batch, linestyle='--', marker='+',
         color='r', label='Batch gradient descent')
ax1.plot(x, y_stochastic, linestyle='--', marker='x',
         color='g', label='Stochastic gradient descent')
ax1.set(xlabel='x', ylabel='y',
        title='Linear regression results')
ax1.legend()
ax1.text(4.2, 5.4, r'$a=$' + str(a_batch)[:4] +
         ', $b=$' + str(b_batch)[:4] + ' and $rmse=$' + str(rmse_batch[-1])[:4], fontsize=10, color='r')
ax1.text(4.2, 4.8, r'$a=$' + str(a_stochastic)[:4] + ', $b=$' + str(b_stochastic)[:4] +
         ' and $rmse=$' + str(rmse_stochastic[-1])[:4], fontsize=10, color='g')
ax2.plot(range(1, len(rmse_batch)+1), rmse_batch, 'r+-', label='Error batch gradient descent')
ax2.plot(range(1, len(rmse_stochastic)+1), rmse_stochastic, 'gx-', label='Error stochastic gradient descent')
ax2.set(xlabel='number pass', ylabel='RMSE',
        title='Error prediction for each pass')
ax2.legend()
plt.show()
