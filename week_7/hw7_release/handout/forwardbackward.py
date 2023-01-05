import argparse
import numpy as np
import math

def parse_args() -> tuple:
    """
    Collects all arguments from command line and returns data. Returns:

        (1) validation_data : A list [X1, X2, ..., XN], where each Xi is a validation 
            example:

                Xi = [(word1, tag1), (word2, tag2), ..., (wordM, tagM)]
            
            This function automatically includes <START> and <END> tags for you.
            
        (2) words_dict : A dictionary mapping words (str) to indices (int).

        (3) tags_dict : A dictionary mapping tags (str) to indices (int).

        (4) emit : A numpy matrix containing the emission probabilities.

        (5) trans : A numpy matrix containing the transition probabilities.

        (6) prediction_file : A string indicating the path to write predictions to.

        (7) metric_file : A string indicating the path to write metrics to.
    
    Usage:
        validation_data, words_dict, tags_dict, emit, trans, prediction_file, metric_file = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('validation_input', type=str,
                        help='path to validation input .txt file')
    parser.add_argument('index_to_word', type=str,
                        help='path to index_to_word.txt file')
    parser.add_argument('index_to_tag', type=str,
                        help='path to index_to_tag.txt file')
    parser.add_argument('emit', type=str,
                        help='path to the learned hmmemit.txt file')
    parser.add_argument('trans', type=str,
                        help='path to the learned hmmtrans.txt file')
    parser.add_argument('prediction_file', type=str,
                        help='path to store predictions')
    parser.add_argument('metric_file', type=str,
                        help='path to store metrics')
    
    args = parser.parse_args()

    # Create train data
    validation_data = list()
    with open(args.validation_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for i in range(len(examples)):
            example = examples[i].split("\n")
            validation_data.append([t.split("\t") for t in example])
    validation_data = [[(None, "<START>")] + elem + [(None, "<END>")] for elem in validation_data]

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
    
    emit = np.loadtxt(args.emit, delimiter=" ")

    trans = np.loadtxt(args.trans, delimiter=" ")

    return validation_data, words_dict, tags_dict, emit, trans, args.prediction_file, args.metric_file


def logsumexp(vi_vector):
    """
    Computes log (sum over all i (e^{x_i})) by using the log-sum-exp trick. You 
    may find it helpful to define a logsumexp function for a matrix X as well. 
    Please note that, when all elements of the vector x are -np.inf, your 
    logsumexp function should return -np.inf and not np.nan.

    Arguments:

        vi_vector : A numpy array of dimension 1 (e.g. a vector, or a list)
    """
    raise NotImplementedError
    

def forwardbackward(x, logtrans, logemit, words_dict, tags_dict):
    """
    Your implementation of the forward-backward algorithm. Remember to compute all 
    values in log-space and use the log-sum-exp trick!

    Arguments:

        x : A list of words

        logtrans : The log of the transition matrix

        logemit : The log of the emission matrix

        words_dict : A dictionary mapping words to indices

        tags_dict : A dictionary mapping tags to indices

    Returns:

        Your choice! The reference solution returns a list containing the predicted 
        tags for each word in x and the log-probability of x.
    
    """
    #forward
    alpha = np.zeros((1,len(tags_dict)), dtype = float)
    alpha[0][0] = 1
    alpha = np.log([alpha[0]])
    # print(alpha)
    for i in range(1, len(x)):
        # print(i)
        new_alpha_list = np.zeros(1, dtype = float)
        for j in range(len(logtrans)):
            if i == len(x)-1:
                logA_j_xj = 0
                # print("logA_",j+1,x[i],": ", logA_j_xj)
            else:
                logA_j_xj = logemit[j][words_dict[x[i]]]
                # print("logA_",j+1,x[i],": ", logA_j_xj)
            logB_vector = logtrans.transpose()[j]
            # print("alpha_prev: ", alpha[i-1])
            # print("logB_vector: ", logB_vector)
            vi_vector = np.add(alpha[i-1], logB_vector)
            max_vi = np.max(vi_vector)
            if max_vi == -np.inf:
                max_vi = 0
            # print("vi_vector: ", vi_vector)
            # print("max_vi:", max_vi)
            # vi_vector = np.delete(vi_vector, np.where(vi_vector == max_vi))
            vi_vector = np.subtract(vi_vector, max_vi)
            # print("vi_vector: ", vi_vector)
            exp_vi = np.exp(vi_vector)
            # print("exp_log_alpha_beta: ", exp_vi)
            log_sum_exp = np.log(np.sum(exp_vi)) + max_vi
            # print("log_sum_exp: ", log_sum_exp, "\n")
            new_alpha_list = np.hstack((new_alpha_list, (logA_j_xj + log_sum_exp)))#np.log(sum_exp) + max_vi)))
        new_alpha_list = new_alpha_list[1:]
        # print("i:",i,"\n", np.exp(new_alpha_list))
        alpha = np.vstack((alpha, new_alpha_list))
    alpha = alpha.transpose()
    # print(np.exp(alpha))
        
    #backward
    beta = np.zeros((1,len(tags_dict)), dtype = float)
    beta[0][len(beta[0])-1] = 1
    beta = np.log([beta[0]])
    # print("beta:\n", beta)
    for i in range(len(x), 1, -1):
        # i_rev = (len(x) - i)
        # print("i: ", i)
        new_beta_list = np.zeros(1, dtype = float)
        for j in range(len(logtrans)):
            log_B_vector = logtrans[j]
            # print("log_B_vector:\t", (log_B_vector))
            log_beta_vector = beta[0]
            # print("log_beta_vec:\t", np.exp(log_beta_vector))
            if i == len(x):
                log_A_vector = np.zeros((len(logtrans)))
            else:
                # print(np.exp(logemit.transpose()[words_dict[x[i-1]]]))
                # print(x[i-1])
                # print(j)
                log_A_vector = logemit.transpose()[words_dict[x[i-1]]].copy()
            # print("log_A_vector:\t", np.exp(log_A_vector))

            vi_vector = np.add(log_B_vector, log_beta_vector)
            vi_vector = np.add(vi_vector, log_A_vector)
            # print("vi_vector:\t", vi_vector)
            max_vi = np.max(vi_vector)
            # print("max_vi: ", max_vi)
            if max_vi == -np.inf:
                max_vi = 0
            vi_vector = np.subtract(vi_vector, max_vi)
            # print("vi_vector:\t", vi_vector)
            exp_vi = np.exp(vi_vector)
            log_sum_exp = np.log(np.sum(exp_vi)) + max_vi
            # print("log_sum_exp:\t", log_sum_exp, "\n")
            new_beta_list = np.hstack((new_beta_list, log_sum_exp))#np.log(sum_exp) + max_vi)))
            # print(np.exp(new_beta_list), "\n")
        new_beta_list = new_beta_list[1:]
        # print("new_beta_list:\t", np.exp(new_beta_list))
        beta = np.vstack((new_beta_list, beta))
    beta = beta.transpose()
    # print(np.exp(beta))


    predicted_tags = []
    log_sum_list = []

    for t in range(len(x)):
        alpha_i = alpha.transpose()[t]
        beta_i  = beta.transpose()[t]
        alpha_i_beta_i = np.add(alpha_i, beta_i)

        log_P_x = alpha[len(alpha)-1][len(alpha[0])-1]
        P_Y_given_x = np.subtract(alpha_i_beta_i, log_P_x)
        predicted_tags.append(np.argmax(np.exp(P_Y_given_x)))

        max_vi = np.max(alpha_i_beta_i)
        if max_vi == -np.inf:
                max_vi = 0
        alpha_i_beta_i = np.subtract(alpha_i_beta_i, max_vi)
        log_sum = np.log(np.sum(np.exp(alpha_i_beta_i))) + max_vi
        log_sum_list.append(log_sum)
    return(predicted_tags, np.sum(log_sum_list)/len(log_sum_list))

    # for t in range(len(x)):
    #     alpha_i = np.exp(alpha.transpose())[t]
    #     beta_i  = np.exp(beta.transpose())[t]
    #     alpha_i_beta_i = np.multiply(alpha_i, beta_i)

        
    #     P_Y_given_x = np.divide(alpha_i_beta_i, np.exp(alpha)[len(alpha)-1][len(alpha[0])-1])
    #     predicted_tags.append(np.argmax(P_Y_given_x))

    #     log_sum = np.log(np.sum(alpha_i_beta_i))
    #     log_sum_list.append(log_sum)
    # return(predicted_tags, np.sum(log_sum_list)/len(log_sum_list))



if __name__ == '__main__':

    validation_data, words_dict, tags_dict, emit, trans, predict_file, metric_file = parse_args()
    tags_list = np.empty(len(tags_dict), dtype = str)
    # for tag in tags_dict:
    #     tags_list[tags_dict[tag]] = tag
    # print(tags_dict)
    tags_dict_reverse = {v: k for k, v in tags_dict.items()}

    logemit = np.log(emit)
    logtrans = np.log(trans)
    # print("logemit:\n", logemit)
    # print("logtrans:\n", logtrans)

    # Iterate over the sentences; for each list of words x, compute its most likely 
    # tags and its log-probability using your forwardbackward function
    total_correct = 0
    total_labels  = 0
    log_likelihood_list = []

    sequence_number = 1

    with open(predict_file, 'w') as f:
        for sentence in validation_data: # sentence looks like [(word1, tag1), (word2, tag2), ..., (wordM, tagM)]

            x = np.array([word for word, tag in sentence]) # x looks like [word1, word2, ..., wordM]
            predicted, log_likelihood = forwardbackward(x, logtrans, logemit, words_dict, tags_dict)
            log_likelihood_list.append(log_likelihood)
            print("sequence" + str(sequence_number) + ": " + str(np.average(np.array(log_likelihood_list))))

            for i in range(1, len(x)-1):
                f.write(x[i] + '\t' + tags_dict_reverse[predicted[i]] + '\n')
            f.write('\n')

            y = np.array([tag for word, tag in sentence]) # y looks like [tag1, tag2, ..., tagM]
            total_labels += len(y)-2
            for i in range(1, len(y)-1):
                if predicted[i] == tags_dict[y[i]]:
                    total_correct += 1
            
            sequence_number += 1

    # Compute the average log-likelihood of all x and the accuracy of your 
    # HMM. When computing the accuracy, you should *NOT* include the first and 
    # last tags, since these are always <START> and <END>. If you're using the 
    # code above, this means that you should only consider y[1:-1] when computing 
    # the accuracy. The accuracy is computed as the total number of correct tags 
    # across all validation sentences divided by the total number of tags across 
    # all validation sentences.

    log_likelihood_list = np.array(log_likelihood_list, dtype = float)


    accuracy = total_correct/total_labels
    avg_LL = np.average(log_likelihood_list)

    # Write the predictions (as words and tags, not indices) and the metrics. 
    # The reference solution doesn't use any special formatting when writing 
    # the metrics.

    with open(metric_file, 'w') as f: #6
        f.write("Average Log-Likelihood: " + str(avg_LL) + '\n')
        f.write("Accuracy: " + str(accuracy) + '\n')
