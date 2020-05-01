"""
Tutorial Simple Linear Regression

Author: Sophie Marchand
"""
import matplotlib as matplot
import matplotlib.pyplot as plt
import numpy as np

# Create a set of points (x,y)
x = np.array([2, 3, 5, 6, 7])
y = np.array([4, 6, 7, 9, 10])

# Compute the means, standard deviations and correlation
x_mean = np.mean(x)
y_mean = np.mean(y)
x_std = np.std(x)
y_std = np.std(y)
xy_corr = np.corrcoef(x, y)[0][1]

# Compute the slop b and the intercept a with means
b_from_mean = sum((x - x_mean) * (y - y_mean)) / sum(np.square(x - x_mean))
a_from_mean = y_mean - b_from_mean * x_mean

# Compute the slop b and the intercept a with standard deviations and correlation
b_from_corr = (xy_corr * y_std) / x_std
a_from_corr = y_mean - b_from_corr * x_mean

# Compute predicted values
y_predicted_from_mean = a_from_mean + b_from_mean * x
y_predicted_from_corr = a_from_corr + b_from_corr * x

# Compute Root Mean Square Error
error_from_mean = np.sqrt(np.sum(np.square(y_predicted_from_mean - y)) / len(y))
error_from_corr = np.sqrt(np.sum(np.square(y_predicted_from_corr - y)) / len(y))

# Print results
fig, ax = plt.subplots()
ax.plot(x, y, 'bo', label='Initial data')
ax.plot(x, y_predicted_from_mean, linestyle='-', marker=matplot.markers.CARETDOWN,
        color='r', label='Mean method')
ax.plot(x, y_predicted_from_corr, linestyle='--', marker=matplot.markers.CARETUP,
        color='g', label='Correlation method')
ax.set(xlabel='x', ylabel='y',
       title='Simple linear regression with two methods:\n mean and correlation')
ax.legend()
ax.text(4.2,5,r'$RMSE_{mean}=RMSE_{correlation}=$'+str(error_from_mean)[:4], fontsize=10)
plt.show()
