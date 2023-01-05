import numpy as np
import sys
import math

#convert .tsv to np_array
def convert_to_np_array(input_file):
    input_file_array = np.genfromtxt(input_file, delimiter = '\t')
    input_file_array = input_file_array[1:]
    return input_file_array

#return entropy float
def calc_entropy(zeros, ones, total):
    probability_zeros = float(zeros/total)
    probability_ones = float(ones/total)
    p0_log2_p0 = probability_zeros * math.log(probability_zeros, 2)
    p1_log2_p1 = probability_ones  * math.log(probability_ones,  2)
    return -1 * (p0_log2_p0 + p1_log2_p1)

#returns majority vote and entropy
def majorityVote_and_entropy(training_file, rows, cols):
    zeros = 0
    ones = 0

    #parse data points for results
    for line in range(rows):
        if training_file[line][cols - 1] == 0:
            zeros += 1
        else: 
            assert(training_file[line][cols - 1] == 1)
            ones += 1

    #find majority
    if zeros > ones: majority_vote = 0
    else:
        assert(zeros <= ones)
        majority_vote = 1
    entrpy = calc_entropy(zeros, ones, rows)
    return majority_vote, entrpy

#return error float
def calc_error(majority_vote, np_array, rows, cols):
    error = 0
    for line in range(rows):
        if majority_vote != np_array[line][cols - 1]: error += 1
    error_rate = float(error)/float(rows)
    return error_rate

if __name__ == '__main__':
    train_file = convert_to_np_array(sys.argv[1])               #input np array
    train_rows, train_cols = train_file.shape                   #input np shape
    majority_vote, entropy = majorityVote_and_entropy(train_file, train_rows, train_cols)
    error = calc_error(majority_vote, train_file, train_rows, train_cols)

    with open(sys.argv[2], 'w') as f:
        f.write("entropy: " + str(entropy) + '\n')
        f.write("error: " + str(error) + '\n')

# train_file = convert_to_np_array("small_train.tsv")
# train_rows, train_cols = train_file.shape
# majority_vote, entropy = majorityVote_and_entropy(train_file, train_rows, train_cols)
# error = calc_error(majority_vote, train_file, train_rows, train_cols)
# print(error)