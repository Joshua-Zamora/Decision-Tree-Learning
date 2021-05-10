import random
import numpy as np

from pprint import pprint

np.seterr(invalid='ignore')


def entropy(probability):
    total = 0
    if probability:
        total += probability * np.log2(probability)
    if probability != 1:
        total += (1 - probability) * np.log2(1 - probability)

    return -total


def remainder(examples, attribute, max_values):
    total = 0
    subsets = np.array([[0, 0]] * max_values[attribute])

    for i in range(examples.shape[0]):
        subsets[examples[i, attribute], examples[i, -1]] += 1

    for i in range(len(subsets)):
        positives = subsets[i, 1]
        negatives = subsets[i, 0]
        total += ((positives + negatives) / examples.shape[0]) * entropy(positives / (positives + negatives))

    return total


def gain(length, examples, max_values):
    gains = np.zeros(length)
    probability = np.count_nonzero(examples[:, -1]) / examples.shape[0]
    b = entropy(probability)

    for i in range(length):
        gains[i] = b - remainder(examples, i, max_values)

    return gains


def importance(attributes, examples, max_values):
    return np.argmax(gain(len(attributes), examples, max_values))


def plurality_value(examples):
    yes_count = np.count_nonzero(examples[:, -1])
    no_count = examples.shape[0] - yes_count

    if yes_count > no_count:
        return "Yes"
    elif yes_count < no_count:
        return "No"
    else:
        return random.choice(["Yes", "No"])


def decision_tree_learning(examples, attributes, parent_examples, max_values):
    yes_count = np.count_nonzero(examples[:, -1])
    no_count = examples.shape[0] - yes_count

    if examples.shape[0] == 0:
        return plurality_value(parent_examples)
    elif yes_count == examples.shape[0]:
        return "Yes"
    elif no_count == examples.shape[0]:
        return "No"
    elif not attributes:
        return plurality_value(examples)
    else:
        index = importance(attributes, examples, max_values)
        tree = [attributes.pop(index)]
        print(index)
        for i in range(max_values.pop(index)):
            sub_examples = examples[examples[:, index] == i]
            sub_examples = np.delete(sub_examples, index, 1)
            sub_tree = decision_tree_learning(sub_examples, attributes, examples, max_values)
            tree.append(sub_tree)

    return tree


def main():
    examples = np.array([
        [1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
        [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
        [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
        [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
        [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1],
    ])

    attributes = ["Alternate", "Bar", "Fri", "Hungry", "Patrons", "Price",
                  "Raining", "Reservation", "Type", "WaitEstimate"]

    max_values = [2, 2, 2, 2, 3, 3, 2, 2, 4, 4]
    tree = decision_tree_learning(examples, attributes, examples, max_values)

    pprint(tree)


if __name__ == "__main__":
    main()
