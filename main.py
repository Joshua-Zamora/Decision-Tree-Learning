import random
from pprint import pprint
from turtle import *
import numpy as np

from draw import draw_tree

max_values = [2, 2, 2, 2, 3, 3, 2, 2, 4, 4]


def remainder(examples, attribute):
    total = 0
    subsets = [[0, 0]] * max_values[attribute]

    for i in range(examples.shape[0]):
        subsets[examples[i, attribute]][examples[i, -1]] += 1

    for i in range(len(subsets)):
        positives = subsets[i][1]
        negatives = subsets[i][0]
        total += ((positives + negatives) / examples.shape[0]) * (positives / (positives + negatives))

    return total


def gain(length, examples):
    gains = np.zeros(length)
    entropy = np.count_nonzero(examples[:, -1]) / (examples.shape[0])

    for i in range(gains.shape[0]):
        gains[i] = abs(entropy - remainder(examples, i))

    return gains


def importance(attributes, examples):
    index = np.argmax(gain(len(attributes), examples))

    return index, attributes[index]


def plurality_value(examples):
    yes_count = np.count_nonzero(examples[:, -1])
    no_count = examples.shape[0] - yes_count

    if yes_count > no_count:
        return "Yes"
    elif yes_count < no_count:
        return "No"
    else:
        return random.choice(["Yes", "No"])


def decision_tree_learning(examples, attributes, parent_examples):
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
        index, a = importance(attributes, examples)
        attributes.pop(index)
        tree = [a]

        for i in range(max_values[index]):
            sub_examples = examples[examples[:, index] == i]
            sub_examples = np.delete(sub_examples, index, 1)
            sub_tree = decision_tree_learning(sub_examples, attributes, examples)
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
        [1, 1, 1, 1, 1, 0, 0, 0, 3, 2, 1],
    ])

    attributes = ["Alternate", "Bar", "Fri", "Hungry", "Patrons", "Price",
                  "Raining", "Reservation", "Type", "WaitEstimate"]

    tree = decision_tree_learning(examples, attributes, examples)

    pprint(tree)
    '''
    up()
    draw_tree(tree, (0, 200), 1)
    done()
    '''


if __name__ == "__main__":
    main()
