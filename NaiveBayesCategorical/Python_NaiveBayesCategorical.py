"""
Tutorial Naive Bayes Categorical
Explanations at: https://sophiemarchand.netlify.app/
Author: Sophie Marchand
"""
import numpy as np
import pandas as pd 
import itertools as it


# Create data for classification of rabbit breeds: French Loop (FL), Jersey Wooly (JW) and Silver Marten (SM)
fur_type = ['short', 'short', 'short', 'long', 'long', 'long', 'long', 'short', 'short', 'short']
ear_type = ['lop', 'lop', 'lop', 'straight', 'straight', 'straight', 'straight', 'straight', 'straight', 'straight']
breed_type = ['French Loop', 'French Loop', 'French Loop', 'Jersey Wooly', 'Jersey Wooly', 'Jersey Wooly', 'Jersey Wooly', 'Silver Marten', 'Silver Marten', 'Silver Marten']
train_data = pd.DataFrame({'fur': fur_type, 'ear': ear_type, 'breed': breed_type})

# Preprocessing to transform categories into numerical number
train_data = train_data.apply(lambda x: pd.factorize(x)[0])

# Compute class probabilities
n_tot = train_data.shape[0]
p_FL = train_data[train_data.breed == 0].shape[0] / n_tot
p_JW = train_data[train_data.breed == 1].shape[0] / n_tot
p_SM = train_data[train_data.breed == 2].shape[0] / n_tot
p_class = [p_FL, p_JW, p_SM]

# Compute conditional probabilities
attribute_label = [0,1]
categories_label = [0,1]
class_label = [0,1,2]
list_conditional_p = []
for attribute, category, class_val in it.product(attribute_label, categories_label, class_label):
    n_class_size = train_data[(train_data.breed == class_val)].shape[0]
    p_inter = train_data[(train_data.iloc[:,attribute] == category) & (train_data.breed == class_val)].shape[0] / n_class_size
    list_conditional_p.append({'attribute': attribute, 'category': category, 'class': class_val, 'probability': p_inter})

# Make predictions 
class_prediction = []
for row in range(train_data.shape[0]):
    p_store = []
    for class_val in class_label:
        dict_fur = next(item for item in list_conditional_p \
            if (item["attribute"], item["category"], item["class"]) \
                == (0, train_data.iloc[row,0], class_val))
        p_fur = dict_fur['probability']
        dict_ear = next(item for item in list_conditional_p \
            if (item["attribute"], item["category"], item["class"]) \
                == (1, train_data.iloc[row,1], class_val))
        p_ear = dict_ear['probability']
        p_store.append(p_fur * p_ear * p_class[class_val])
    class_prediction.append(p_store.index(max(p_store)))

# Compute accuracy
accuracy = sum(train_data.iloc[:,2] == class_prediction) / n_tot
