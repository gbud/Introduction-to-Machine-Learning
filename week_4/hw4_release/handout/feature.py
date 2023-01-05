import csv
import numpy as np
import sys
import math

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

def one_hot_vector(label, review, dictionary):
    vector = np.zeros(len(dictionary)+1, dtype = int)
    vector[0] = label
    for word in review.split():
        if (word in dictionary):
            index = dictionary[word]
            vector[index + 1] = 1
    return vector

#returns array of trimmed words
def trim(review, dictionary):
    result = []
    for word in review.split():
        if (word in dictionary):
            result.append(word)
    return result

def word2vec(label, review, dictionary):
    vector = np.zeros(300, dtype = float)
    for word in review:
        vector += dictionary[word]
    vector /= len(review)
    label_array = np.zeros(1, dtype = float)
    label_array[0] = label
    result = np.concatenate((label_array, vector),axis = 0)
    return result

def model1_output(input_file, dictionary, output_file):
    train_output_matrix = np.zeros(len(dictionary)+1, dtype = int)
    for data_point in input_file:
        data_label, data_review = data_point
        new_row = one_hot_vector(data_label, data_review, dictionary)
        train_output_matrix = np.vstack((train_output_matrix, new_row))
    train_output_matrix = train_output_matrix[1:]
    np.savetxt(output_file, train_output_matrix.astype(int), 
                fmt = '%i', delimiter = "\t", newline = "\n", )

def model2_output(input_file, feature_dictionary, output_file):
    train_output_matrix = np.zeros(301, dtype = float)
    for data_point in input_file:
        data_label, data_review = data_point
        trim_review = trim(data_review, feature_dictionary)
        new_row = word2vec(data_label, trim_review, feature_dictionary)
        train_output_matrix = np.vstack((train_output_matrix, new_row))
    train_output_matrix = train_output_matrix[1:]
    train_output_matrix = np.around(train_output_matrix, decimals = 6)
    np.savetxt(output_file, train_output_matrix.astype(float), 
                fmt = '%f', delimiter = "\t", newline = "\n", )

if __name__ == '__main__':
    train_file = load_tsv_dataset(sys.argv[1])                          #arg1
    validation_file = load_tsv_dataset(sys.argv[2])                     #arg2
    test_file = load_tsv_dataset(sys.argv[3])                           #arg3
    dictionary = load_dictionary(sys.argv[4])                           #arg4
    feature_dictionary = load_feature_dictionary(sys.argv[5])           #arg5
    flag = int(sys.argv[9])                                             #arg9

    #bag of words algo
    if flag == 1: 
        model1_output(train_file, dictionary, sys.argv[6])
        model1_output(validation_file, dictionary, sys.argv[7])
        model1_output(test_file, dictionary, sys.argv[8])

    #word embeddings algo
    if flag == 2:
        model2_output(train_file, feature_dictionary, sys.argv[6])
        model2_output(validation_file, feature_dictionary, sys.argv[7])
        model2_output(test_file, feature_dictionary, sys.argv[8])
