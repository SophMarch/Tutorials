"""
Tutorial Linear Discriminant Analysis
Explanations at: https://sophiemarchand.netlify.app/
Author: Sophie Marchand
"""
import matplotlib.pyplot as plt
import numpy as np


# Function compute probability class, mean, variance and accuracy
def compute_probability_class(y, class_y):
    return sum(y == class_y) / len(y)


def compute_mean_x_for_class(x, y, class_y):
    return np.mean(x[y == class_y])


def compute_variance(x, y):
    mean = []
    for class_y in np.unique(y):
        mean.append(compute_mean_x_for_class(x, y, class_y))
    mean_1 = [mean[0]] * len(x_1)
    mean_2 = [mean[1]] * len(x_2)
    mean_3 = [mean[2]] * len(x_3)
    mean_vector = np.asarray([*mean_1, *mean_2, *mean_3])
    variance = (1/(len(x) - len(np.unique(y)))) * np.sum(np.square(x - mean_vector))
    return variance


def compute_discriminant_single_variable(x, y, class_y, x_value):
    mean = compute_mean_x_for_class(x, y, class_y)
    variance = compute_variance(x, y)
    discriminant = compute_probability_class(y, class_y) + \
                   (x_value * mean / variance) - (np.square(mean) / (2 * variance))
    return discriminant


def compute_accuracy(y, y_pred):
    return sum(y == y_pred)/len(y)


# Create data for multiclass classification with a single input variable
x_1 = np.random.normal(20, 2, 20)
x_2 = np.random.normal(42, 2, 20)
x_3 = np.random.normal(60, 2, 20)
x = np.asarray([*x_1, *x_2, *x_3])

y_1 = [1]*len(x_1)
y_2 = [2]*len(x_2)
y_3 = [3]*len(x_3)
y = np.asarray([*y_1, *y_2, *y_3])

y_pred = []

# For each single input variable compute the discriminant function for all classes and attribute class
for i in range(len(x)):
    discriminant = []
    for y_class in np.unique(y):
        discriminant_value = compute_discriminant_single_variable(x, y, y_class, x[i])
        discriminant.append(discriminant_value)
    x_class = np.where(discriminant == np.max(discriminant))[0][0]
    y_pred.append(x_class+1)

# Print results
y_pred = np.asarray(y_pred)
y_accuracy = compute_accuracy(y, y_pred)
fig, ax = plt.subplots()
fig.suptitle('Linear Discriminant Analysis for classification problem with 3 classes: accuracy = ' + str(y_accuracy), fontsize=15)
label_legend = ['class 1', 'class 2', 'class 3']
for color in ['g', 'y', 'r']:
    if color == 'g':
        index = np.where(y_pred == 1)
        label = label_legend[0]
    elif color == 'y':
        index = np.where(y_pred == 2)
        label = label_legend[1]
    else:
        index = np.where(y_pred == 3)
        label = label_legend[2]
    print(x[index])
    plt.plot(x[index], '.', markersize=15, c=color, label=label)
ax.set(xlabel='Number of samples', ylabel='$x$')
plt.xticks(np.arange(0, len(x[index]), 2.0))
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
ax.legend()
plt.show()
