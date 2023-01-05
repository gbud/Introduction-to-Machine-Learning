import numpy as np
import sys
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

#returns tuples of X and y
def process_input(input_file):
    input_file_array = np.genfromtxt(input_file, delimiter = '\t', dtype = float)
    input_file_transpose = input_file_array.transpose()
    y_labels = input_file_transpose[0].copy()
    X_features = input_file_transpose[1:]
    bias_fold = np.ones(len(X_features[0]), dtype = float)
    X_features = np.vstack((bias_fold, X_features))
    return X_features.transpose(), y_labels

#returns np.array of predictions
def predict(theta, X):
    predictions = []
    for i in range(len(X)):
        thetaT_xi = np.dot(theta, X[i])
        probability = sigmoid(thetaT_xi)
        if probability >= 0.5:
            predictions.append(1.0)
        else:
            assert(probability < 0.5)
            predictions.append(0.0)
    return np.array(predictions)

#returns float rounded to 6
def compute_error(y_pred, y):
    error = 0
    for i in range(len(y)):
        if y_pred[i] != y[i]: error += 1
    return error/len(y)

def avg_neg_loglikelihood(theta, X, y):
    result = 0
    theta_X_array = np.dot(X, theta)
    for i in range(len(y)):
        ji_theta = -y[i]*theta_X_array[i] + math.log(1 + math.e**theta_X_array[i])
        result += ji_theta
    return result/len(y)

#returns learned weights, theta_list
def train(theta, X, y, num_epoch, learning_rate, val_X, val_y):
    train_theta_list = []
    val_theta_list = []
    for epoch in range(num_epoch):
        for i in range(len(X)):
            thetaT_xi = np.dot(theta, X[i])
            theta += learning_rate * X[i] * (y[i] - sigmoid(thetaT_xi))

        train_avg_likelihood = avg_neg_loglikelihood(theta, X, y)
        train_theta_list.append(train_avg_likelihood)

        val_avg_likelihood = avg_neg_loglikelihood(theta, val_X, val_y)
        val_theta_list.append(val_avg_likelihood)

    train_theta_list = np.array(train_theta_list)
    val_theta_list = np.array(val_theta_list)

    return theta, train_theta_list, val_theta_list

if __name__ == '__main__':
    formatted_train_input = process_input(sys.argv[1])  #1
    formatted_validation_input = process_input(sys.argv[2]) #2
    formatted_test_input = process_input(sys.argv[3]) #3
    num_epochs = int(sys.argv[7]) #sys.argv[7]
    learning_rate = float(sys.argv[8]) #sys.argv[8]

    #process inputs
    train_X, train_y = formatted_train_input
    val_X, val_y = formatted_validation_input
    test_X, test_y = formatted_test_input

    #learn weights
    theta_init = np.zeros(len(train_X[0]), dtype = float)
    theta_train, training, validation = train(theta_init, 
                                    train_X, train_y, num_epochs, learning_rate,
                                    val_X, val_y)

    np.savetxt(sys.argv[9], validation.astype(float), 
                fmt = '%f', delimiter = "\n")

    #predict sets
    predict_train = predict(theta_train, train_X)
    predict_val = predict(theta_train, val_X)
    predict_test = predict(theta_train, test_X)

    #save labels
    np.savetxt(sys.argv[4], predict_train.astype(int), 
                fmt = '%i', delimiter = "\n") #4
    np.savetxt(sys.argv[5], predict_test.astype(int), 
                fmt = '%i', delimiter = "\n") #5

    #calc errors
    train_error = compute_error(predict_train, train_y)
    test_error = compute_error(predict_test, test_y)

    with open(sys.argv[6], 'w') as f: #6
        f.write("error(train): " + str(train_error) + '\n')
        f.write("error(test): " + str(test_error) + '\n')

    #model3
    formatted_train_input = np.loadtxt("model3_val_nll.txt").transpose()[1:]
    formatted_train_input = formatted_train_input[0]

    np.savetxt("largedata/model3_train_theta_list.tsv", 
               formatted_train_input.astype(float), fmt = '%f', delimiter = "\n")

    #plot j_theta_list
    x = []
    for i in range(int(sys.argv[7])):
        x.append(i+1)
    plt.plot(x, training)
    plt.plot(x, validation)
    # plt.plot(x, formatted_train_input)
    plt.xlabel("Epochs")
    plt.ylabel("Average Negative Likelihood")
    plt.title("Model 1 Large Dataset")
    plt.legend(["training","validation"])
    plt.show()