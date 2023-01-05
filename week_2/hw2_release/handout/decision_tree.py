import numpy as np
import math
import sys
import matplotlib.pyplot as plt

def attribute_list(input_file):
    input_file_array = np.genfromtxt(input_file, delimiter = '\t', dtype = str)
    input_file_array = input_file_array[:1]
    return input_file_array[0]

def convert_to_np_array(input_file):
    input_file_array = np.genfromtxt(input_file, delimiter = '\t', dtype = int)
    input_file_array = input_file_array[1:]
    return input_file_array

def calc_zeros_and_ones(np_array, attribute):
    rows = len(np_array)
    zeros = 0
    ones = 0
    for line in range(rows):
        if np_array[line][attribute] == 0:
            zeros += 1
        else: 
            assert(np_array[line][attribute] == 1)
            ones += 1
    return zeros, ones

def calc_majority_vote(zeros, ones):
    if zeros > ones: majority_vote = 0
    else:
        assert(zeros <= ones)
        majority_vote = 1
    return majority_vote

#returns np array tuple of split at attribute 0's vs 1's
def split_array(np_array, attribute):
    np_array_zeros = []
    np_array_ones = []

    for i in range(len(np_array)):
        if np_array[i][attribute] == 0: np_array_zeros.append(np_array[i])
        else:
            assert(np_array[i][attribute] == 1)
            np_array_ones.append(np_array[i])

    return np.array(np_array_zeros), np.array(np_array_ones)

def calc_entropy(np_array, attribute):
    zeros, ones = calc_zeros_and_ones(np_array, attribute)
    if zeros == 0 or ones == 0: return 0
    rows = len(np_array)
    probability_zeros = float(zeros/rows)
    probability_ones = float(ones/rows)
    p0_log2_p0 = probability_zeros * math.log(probability_zeros, 2)
    p1_log2_p1 = probability_ones  * math.log(probability_ones,  2)
    return -1 * (p0_log2_p0 + p1_log2_p1)

def calc_mutual_information(np_array, attribute):
    rows, cols = np_array.shape
    attr_zeros,  attr_ones  = calc_zeros_and_ones(np_array, attribute)

    pX0 = float(attr_zeros/rows)
    pX1 = float(attr_ones /rows)

    np_array_X0, np_array_X1 = split_array(np_array, attribute)


    return (calc_entropy(np_array, cols - 1) -
           (pX0 * calc_entropy(np_array_X0, cols - 1)) - 
           (pX1 * calc_entropy(np_array_X1, cols - 1)))


def highest_mutual_information(np_array):
    best_attribute = None
    best_MI = None
    cols = len(np_array[0])
    for i in range(cols - 1):
        if calc_entropy(np_array, i) == 0: continue         #skips pure data
        else:
            curr_MI = calc_mutual_information(np_array, i)
            if best_MI == None or curr_MI > best_MI: 
                best_MI = curr_MI
                best_attribute = i
    return best_attribute

class Node:
    def __init__(self, data, depth, attribute_list):
        self.left = None                    #left child
        self.right = None                   #right child
        self.attribute = None               #feature node is splitting upon
        self.data = data                    #numpy array
        self.depth = depth                  #node depth
        self.majority_vote = None
        self.rows, self.cols = data.shape
        self.attribute_list = attribute_list

    def train(self, max_depth):
        #contracts
        assert(max_depth < self.cols)

        #if BASECASE
            #if node depth > max, label entropy is pure, or max MI is 0...
        if ((self.depth > max_depth) or (calc_entropy(self.data, len(self.data[0])-1) == 0) or
            (calc_mutual_information(self.data, highest_mutual_information(self.data)) == 0)):
            label_zeros, label_ones = calc_zeros_and_ones(self.data, self.cols - 1)
            self.majority_vote = calc_majority_vote(label_zeros, label_ones)

        #if RECURRSIVE CASE
        else:
            #1. find feature to split on 
            self.attribute = highest_mutual_information(self.data)

            #2. split
            split_zeros, split_ones = split_array(self.data, self.attribute)
            self.left = Node(split_zeros, self.depth+1, self.attribute_list)
            self.right = Node(split_ones, self.depth+1, self.attribute_list)

            #3. train recurrsively
            self.left.train(max_depth)
            self.right.train(max_depth)

def predict(node, data_point):
    #basecase
    if node.left == None and node.right == None:
        return node.majority_vote

    #recurrsive case
    else:
        if data_point[node.attribute] == 0:
            return predict(node.left, data_point)
        else:
            assert(data_point[node.attribute] == 1)
            return predict(node.right, data_point)

def calc_error(prediction_txt, np_array, rows, cols):
    prediction_array = np.genfromtxt(prediction_txt)
    error = 0
    for line in range(rows):
        if prediction_array[line] != np_array[line][cols - 1]: error += 1
    error_rate = float(error)/float(rows)
    return error_rate

def print_tree(node, flag):
    if flag == True: #print root entropy if true
        dt_zeros, dt_ones = calc_zeros_and_ones(node.data, node.cols - 1)
        print("[" + str(dt_zeros) + " 0/" + str(dt_ones) + " 1]")
    if node.left == None and node.right == None: return
    else:
        split_zeros, split_ones = split_array(node.data, node.attribute)
        zeros_zeros, zeros_ones = calc_zeros_and_ones(split_zeros, node.cols - 1)
        print("| "*node.depth + node.attribute_list[node.attribute] + " = 0: [" + str(zeros_zeros) + " 0/" + str(zeros_ones) + " 1]")
        print_tree(node.left, False)
        ones_zeros, ones_ones = calc_zeros_and_ones(split_ones, node.cols - 1)
        print("| "*node.depth + node.attribute_list[node.attribute] + " = 1: [" + str(ones_zeros)  + " 0/" + str(ones_ones) + " 1]")
        print_tree(node.right, False)

        
if __name__ == '__main__':
    train_file = convert_to_np_array(sys.argv[1])
    attribute_list = attribute_list(sys.argv[1])
    decision_tree = Node(train_file, 1, attribute_list)            #init dc_tree

# for j in range(len(attribute_list)-1):

    decision_tree.train(int(sys.argv[3]))
    print_tree(decision_tree, True)     #true flag to print title

    #write train prediction file
    with open(sys.argv[4], 'w') as f:
        for i in range(len(train_file)):
            out = predict(decision_tree, train_file[i])
            f.write(str(out) + '\n')

    #write test prediction file
    test_file = convert_to_np_array(sys.argv[2])
    with open(sys.argv[5], 'w') as f:
        for i in range(len(test_file)):
            out = predict(decision_tree, test_file[i])
            f.write(str(out) + '\n')

    train_error = calc_error(sys.argv[4], train_file, len(train_file), len(train_file[0]))
    test_error = calc_error(sys.argv[5], test_file, len(test_file), len(test_file[0]))
    # print(str(j) + ":\t" + str(train_error) + "\t" + str(test_error))

    with open(sys.argv[6], 'w') as f:
        f.write("error(train): " + str(train_error) + '\n')
        f.write("error(test): " + str(test_error) + '\n')