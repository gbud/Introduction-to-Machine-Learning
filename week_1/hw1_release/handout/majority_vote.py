import numpy as np
import sys

#convert .tsv to np_array
def convert_to_np_array(input_file):
    input_file_array = np.genfromtxt(input_file, delimiter = '\t')
    input_file_array = input_file_array[1:]
    return input_file_array

#returns majority vote
def train(training_file, rows, cols):
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
    return majority_vote

#writes predictions on label file
def predict(label_file, rows, prediction):
    with open(label_file, 'w') as f:
        for i in range(rows):
            f.write(str(prediction) + '\n')
    return #void

#returns error float
def calc_error(prediction_txt, np_array, rows, cols):
    prediction_array = np.genfromtxt(prediction_txt)
    error = 0
    for line in range(rows):
        if prediction_array[line] != np_array[line][cols - 1]: error += 1
    error_rate = float(error)/float(rows)
    return error_rate

if __name__ == "__main__":
    train_file = convert_to_np_array(sys.argv[1])               #input np array
    train_rows, train_cols = train_file.shape                   #input np shape
    majority_vote = train(train_file, train_rows, train_cols)   #train input

    test_file = convert_to_np_array(sys.argv[2])                #test np array
    test_rows, test_cols = test_file.shape                      #test np shape

    predict(sys.argv[3], train_rows, majority_vote)             #predict train
    predict(sys.argv[4], test_rows, majority_vote)              #predict test

    train_error = calc_error(sys.argv[3], train_file, train_rows, train_cols)
    test_error = calc_error(sys.argv[4], test_file, test_rows, test_cols)

    # train_error = format(round(train_error, 6), '.6f')
    # test_error = format(round(test_error, 6), '.6f')

    with open(sys.argv[5], 'w') as f:
        f.write("error(train): " + str(train_error) + '\n')
        f.write("error(test): " + str(test_error) + '\n')