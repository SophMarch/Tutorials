"""
Tutorial Gaussian Naive Bayes
Explanations at: https://sophiemarchand.netlify.app/
Author: Sophie Marchand
"""
import numpy as np
import pandas as pd 
from scipy.stats import norm
import matplotlib.pyplot as plt
np.random.seed(42) 

# Create data for classification of men and women according to height and weight 
distribution_characteristics = [[175, 163, 77, 64], [7, 7, 8, 8]]
df_dist = pd.DataFrame(distribution_characteristics, \
    columns=['P(height|men)', 'P(height|women)', 'P(weight|men)', 'P(weight|women)'], index=['Mean', 'Std'])
n = 30
men_height_cm = np.random.normal(df_dist.iloc[0,0], df_dist.iloc[1,0], n) 
women_height_cm = np.random.normal(df_dist.iloc[0,1], df_dist.iloc[1,1], n) 
height_cm = np.concatenate((men_height_cm, women_height_cm))

men_weight_kg = np.random.normal(df_dist.iloc[0,2], df_dist.iloc[1,2], n)
women_weight_kg = np.random.normal(df_dist.iloc[0,3], df_dist.iloc[1,3], n) 
weight_cm = np.concatenate((men_weight_kg, women_weight_kg))

gender = [0]*len(men_height_cm) + [1]*len(women_height_cm)
train_data = pd.DataFrame({'height_cm': height_cm, 'weight_cm': weight_cm, 'gender': gender})

# Compute class probabilities
n_tot = train_data.shape[0]
p_men = train_data[train_data.gender == 0].shape[0] / n_tot
p_women = train_data[train_data.gender == 1].shape[0] / n_tot
p_class = [p_men, p_women]

# Make predictions 
class_label = [0, 1]
class_prediction = []
for row in range(train_data.shape[0]):
    p_store = []
    for class_val in class_label:
        x_height = train_data.iloc[row,0]
        x_weight = train_data.iloc[row,1]
        p_height = norm(df_dist.iloc[0,class_val], df_dist.iloc[1,class_val]).pdf(x_height)
        p_weight = norm(df_dist.iloc[0,class_val+2], df_dist.iloc[1,class_val+2]).pdf(x_weight)
        p_store.append(p_height * p_weight * p_class[class_val])
    class_prediction.append(p_store.index(max(p_store)))

# Print results
y_pred = np.asarray(class_prediction)
y_accuracy = sum(train_data.iloc[:,2] == class_prediction) / n_tot
fig, ax = plt.subplots()
fig.suptitle('Gaussian Naive Bayes: accuracy = ' + str(y_accuracy), fontsize=15)
label_legend = ['men', 'women']
for color in ['b', 'g']:
    if color == 'b':
        index = np.where(y_pred == 0)
        label = label_legend[0]
    else:
        index = np.where(y_pred == 1)
        label = label_legend[1]
    plt.plot(train_data.iloc[index[0],0], train_data.iloc[index[0],1], '.', \
        markersize=15, c=color, label=label)
ax.set(xlabel='height in cm', ylabel='weight in kg')
ax.legend(loc='lower right')
plt.show()


