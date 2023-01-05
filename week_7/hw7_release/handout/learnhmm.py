import argparse
import numpy as np

def parse_args() -> tuple:
    """
    Collects all arguments passed in via command line and returns the appropriate 
    data. Returns:

        (1) train_data : A list [X1, X2, ..., XN], where each element Xi is a training 
            example represented as a list of tuples:

                Xi = [(word1, tag1), (word2, tag2), ..., (wordM, tagM)]

            For example:

                train_data = [[(None, "<START>"), ("fish", "D"), (next_tuple)], [next_train_example], ...]

            Note that this function automatically includes the "<START>" and 
            "<END>" tags for you.

        (2) words_dict : A dictionary with keys of type str and values of type int. 
            Keys are words and values are their indices. For example:

                words_dict["hi"] == 99

        (3) tags_dict : A dictionary with keys of type str and values of type int.
            Keys are tags and values are their indices. For example:

                tags_dict["<START>"] == 0

        (4) emit : A string representing the path of the output hmmemit.txt file.
        
        (5) trans : A string representing the path of the output hmmtrans.txt file.
    
    Usage:
        train_data, word_dict, tags_dict, emit, trans = parse_args()
    """
    # Define a parser
    parser = argparse.ArgumentParser()
    parser.add_argument('train_input', type=str,
                        help='path to training input .txt file')
    parser.add_argument('index_to_word', type=str,
                        help='path to index_to_word.txt file')
    parser.add_argument('index_to_tag', type=str,
                        help='path to index_to_tag.txt file')
    parser.add_argument('emit', type=str,
                        help='path to store the hmmemit.txt file')
    parser.add_argument('trans', type=str,
                        help='path to store the hmmtrans.txt file')
    
    args = parser.parse_args()

    # Create train data
    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for i in range(len(examples)):
            example = examples[i].split("\n")
            train_data.append([t.split("\t") for t in example])
    train_data = [[(None, "<START>")] + elem + [(None, "<END>")] for elem in train_data]

    # making dictionary of words to index
    words_dict = {}
    with open(args.index_to_word, "r") as words_indices:
        i = 0
        for line in words_indices:
            words_dict[line.rstrip()] = i
            i += 1

    # making dictionary of words to tags
    tags_dict = {}
    with open(args.index_to_tag, "r") as tags_indices:
        j = 0
        for line in tags_indices:
            tags_dict[line.rstrip()] = j
            j += 1
    
    return train_data, words_dict, tags_dict, args.emit, args.trans


if __name__ == "__main__":
    
    train_data, words_dict, tags_dict, emit, trans = parse_args()

    # Initialize emit (A) and trans (B) matrices
    B = np.ones((len(tags_dict)-1, len(tags_dict)-1), dtype = float)
    B_stack = np.zeros(len(tags_dict)-1, dtype = float)
    B = np.vstack((B, B_stack)).transpose()
    B_stack = np.zeros(len(tags_dict), dtype = float)
    B = np.vstack((B_stack, B)).transpose()
    
    A = np.ones((len(tags_dict), len(words_dict)), dtype = float)

    # Iterate through the data and increment the appropriate cells in the matrices
    for sequence in train_data:
        prev_state = 0
        for i in range(1,len(sequence)-1):
            word_index  = words_dict[sequence[i][0]]
            state_index = tags_dict[sequence[i][1]]
            B[prev_state][state_index] += 1
            A[state_index][word_index] += 1
            prev_state = state_index
        B[prev_state][len(tags_dict)-1] += 1

    # Convert the rows of A and B to probability distributions. Each row 
    # of A and B should sum to 1 (except for the rows mentioned below). 
    # Please note that:
    # 
    #   B[:, tags_dict["<START>"]] == 0 since nothing can transition to <START>
    #   B[tags_dict["<END>"], :]   == 0 since <END> can't transition to anything
    #   A[tags_dict["<START>"], :] == 1 since <START> emits nothing; setting to 1 makes forwardbackward easier (as opposed to setting to 0)
    #   A[tags_dict["<END>"], :]   == 1 since <END> emits nothing
    #
    # You should manually ensure that the four conditions above hold (e.g. by 
    # manually setting the rows/columns to the desired values and ensuring that 
    # the other rows not mentioned above remain probability distributions)

    for i in range(1, len(A)-1):
        row_sum = A[i].sum()
        A[i] = np.divide(A[i],row_sum)

    for i in range(0, len(B)-1):
        row_sum = B[i].sum()
        B[i] = np.divide(B[i],row_sum)

    # Save the emit and trans matrices (the reference solution uses np.savetxt 
    # with fmt="%.18e")
    np.savetxt(emit,  A.astype(float), delimiter = " ", newline = "\n")
    np.savetxt(trans, B.astype(float), delimiter = " ", newline = "\n")