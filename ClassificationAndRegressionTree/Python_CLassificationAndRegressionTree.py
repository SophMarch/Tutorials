"""
Tutorial Linear Discriminant Analysis
Explanations at: https://sophiemarchand.netlify.app/
Credit: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
Adapted by: Sophie Marchand
"""

import numpy as np


# Functions to create and display classification tree
# Create a split within the training data
def create_split_in_data(data, col_variable, split_variable_value):
    split_part_1, split_part_2 = list(), list()
    for row in data:
        if row[col_variable] < split_variable_value:
            split_part_1.append(row)
        else:
            split_part_2.append(row)
    return split_part_1, split_part_2


# Get the Gini index value for a given split
def compute_gini_index_for_split(split_parts, classes):
    tot_size = sum([len(split) for split in split_parts])
    gini_index = 0
    for split in split_parts:
        split_size = len(split)
        if split_size == 0:
            continue
        gini_class_score = 0
        for class_label in classes:
            p = [row[-1] for row in split].count(class_label) / split_size
            gini_class_score += p * p
        gini_index += (1 - gini_class_score) * (split_size / tot_size)
    return gini_index


# Get the best split according to the gini index checking every variable of every attribute
def get_best_split(data):
    classes_val = set([row[-1] for row in data])
    best_col, best_value, best_score, best_splits = 0, 0, 100, None
    for col_variable in range(train_data.shape[1] - 1):
        for row in data:
            split_test = create_split_in_data(data, col_variable, row[col_variable])
            gini = compute_gini_index_for_split(split_test, classes_val)
            # print('X%d < %.3f Gini=%.3f' % ((col_variable + 1), row[col_variable], gini))
            if gini < best_score:
                best_col, best_value, best_score, best_splits = col_variable, row[col_variable], gini, split_test
    return {'index': best_col, 'value': best_value, 'splits': best_splits}


# Stop the splitting by attributing the value of the class
def terminate(split):
    y_in_split = [row[-1] for row in split]
    return max(set(y_in_split), key=y_in_split.count)


# Recursive splitting to create tree
def tree_split(node, max_depth, min_size, depth):
    split_part_1, split_part_2 = node['splits']
    del (node['splits'])
    if not split_part_1 or not split_part_2:
        node['split_part_1'] = node['split_part_2'] = terminate(split_part_1 + split_part_2)
        return
    if depth >= max_depth:
        node['split_part_1'], node['split_part_2'] = terminate(split_part_1), terminate(split_part_2)
        return
    if len(split_part_1) <= min_size:
        node['split_part_1'] = terminate(split_part_1)
    else:
        node['split_part_1'] = get_best_split(split_part_1)
        tree_split(node['split_part_1'], max_depth, min_size, depth + 1)
    if len(split_part_2) <= min_size:
        node['split_part_2'] = terminate(split_part_2)
    else:
        node['split_part_2'] = get_best_split(split_part_2)
        tree_split(node['split_part_2'], max_depth, min_size, depth + 1)


# Initiate the node and implement the splitting
def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    tree_split(root, max_depth, min_size, 1)
    return root


# Visualization of the tree structure
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % (depth * ' ', (node['index'] + 1), node['value']))
        print_tree(node['split_part_1'], depth + 1)
        print_tree(node['split_part_2'], depth + 1)
    else:
        print('%s[%s]' % (depth * ' ', node))


# Create data for classification of apple ripeness
x_diameter = [5.4, 6.8, 7.4, 8.3, 9.4, 9.8, 6.9, 7.9, 7.4, 7.2]
x_weight = [420, 380, 480, 432, 480, 450, 580, 530, 620, 580]
y_ripeness = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
train_data = np.array([[x_diameter[i], x_weight[i], y_ripeness[i]] for i in range(len(x_diameter))])
y_prediction = []

# Go through every input variable and compute Gini index
tree = build_tree(train_data, 2, 1)
print_tree(tree)
