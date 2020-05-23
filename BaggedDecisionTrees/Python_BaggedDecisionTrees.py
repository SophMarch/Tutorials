"""
Tutorial Bagged Decision Tree
Explanations at: https://sophiemarchand.netlify.app/
Credit: https://machinelearningmastery.com/implement-bagging-scratch-python/
Adapted by: Sophie Marchand
"""
from csv import reader
from random import randrange, seed
seed(42)

# Functions to load and process the dataset
# Load dataset
def load_data(filename):
    dataset = list()
    with open(filename, 'r') as file:
        file_reader = reader(file)
        for row in file_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Transform string attributes in fload
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])

# Transform class attribute into integers
def str_column_to_int(dataset, column):
    all_class_values = [row[column] for row in dataset]
    class_categories = set(all_class_values)
    lookup = dict()
    for i, value in enumerate(class_categories):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Functions for the evaluation of the bagging algorithm
# Dataset split
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split 

# Compute accuracy
def accuracy_metric(real, predicted):
    correct = 0
    for i in range(len(real)):
        if real[i] == predicted[i]:
            correct += 1
    return correct / float(len(real)) * 100.0

# Workflow evaluation algorithm
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


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
    for col_variable in range(len(dataset[0]) - 1):
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


# Functions for bagging application
# Get a subsample of the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample 

# Make a prediction on one decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['split_part_1'], dict):
            return predict(node['split_part_1'], row)
        else:
            return node['split_part_1']
    else:
        if isinstance(node['split_part_2'], dict):
            return predict(node['split_part_2'], row)
        else:
            return node['split_part_2']

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# Bootstrap Aggregation ALgorithm
def bagging(train, test, max_depth, min_size, sample_size, n_trees):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)


# Load the sonar dataset and transform datatype
dataset = load_data('BaggedDecisionTree\sonar_dataset.csv')
for id_attribute_col in range(len(dataset[0])-1):
    str_column_to_float(dataset, id_attribute_col)
str_column_to_int(dataset, len(dataset[0])-1)

# Evaluate bagging algorithm
n_folds = 5
max_depth = 6
min_size = 2
sample_size = 0.5 
for n_trees in [1, 5, 10, 15]:
    scores = evaluate_algorithm(dataset, bagging, n_folds, max_depth, min_size, sample_size, n_trees)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))