import numpy as np
import argparse
import logging
import math

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)


def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    # Implement random initialization here
    result = np.random.uniform(low = -0.1, high = 0.1, size = shape)
    return result


def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    result = np.zeros(shape)
    return result


def linear(X, w):
    """
    Implement linear forward
    :param X: input logits of shape (input_size)
    :param w: weights of shape (hidden_size, input_size) or (output_size, hidden_size + 1)
    :return: linear forward output of shape (hidden_size) or (output_size)
    """
    # print(X.shape)
    # print(w.shape)
    return np.matmul(X, w.transpose())


def sigmoid(a):
    """
    Implement sigmoid function.
    :param a: input logits of shape (hidden_size)
    :return: sigmoid output of shape (hidden_size)
    """
    e = np.exp(a)
    return np.hstack((np.ones(1, dtype = float), e / (1 + e)))



def softmax(b):
    """
    Implement softmax function.
    :param b: input logits of shape (output_size)
    :return: softmax output of shape (output_size)
    """
    e = np.exp(b)
    return e / (np.sum(e))


def cross_entropy(y, y_hat):
    """
    Compute cross entropy loss.
    :param y: label
    :param y_hat: prediction
    :return: cross entropy loss
    """
    return -math.log(y_hat[y])


def d_linear(alpha_or_beta, a_or_z, w):
    """
    COmpute gradients of loss w.r.t. linear input and weights
    :param X: input to linear layer
    :param w:
    :return: a tuple (gradient w.r.t. X, gradient w.r.t. w)
    """
    g_A_or_B = np.outer(w, a_or_z)
    g_x_or_z = np.dot(alpha_or_beta.transpose()[1:], w)
    return g_A_or_B, g_x_or_z


def d_sigmoid(Z):
    """
    Compute gradient of sigmoid output w.r.t. its input.
    :param Z: sigmoid's input
    :return: gradient
    """
    # print(Z.z)
    # print(np.ones(len(Z.z)) - Z.z)
    return np.multiply(Z.z,(np.ones(len(Z.z)) - Z.z))[1:].reshape(len(Z.z)-1, 1)


def d_cross_entropy_vec(y, y_hat):
    """
    Compute gradient of loss w.r.t. ** softmax input **.
    Note that here instead of calculating the gradient w.r.t. the softmax probabilities,
    we are directly computing the gradient w.r.t. the softmax input.
    Try derive the gradient yourself, and you'll see why we want to calculate this in a single step
    :param y: label of shape (output_size)
    :param y_hat: predicted softmax probability of shape (output_size)
    :return: gradient of shape (output_size)
    """
    y_vector = np.zeros(len(y_hat), dtype = float)
    y_vector[y] = 1.0
    return (y_hat - y_vector).reshape(len(y_hat), 1)


class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer *including* the folded bias
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size
        self.nn_a = None
        self.nn_b = None
        self.nn_z = None

        # initialize weights and biases for the models
        # HINT: pay attention to bias here
        if weight_init_fn == 1:
            self.alpha = random_init([hidden_size, input_size])
            self.beta  = random_init([output_size, hidden_size + 1])
        if weight_init_fn == 2:
            self.alpha = zero_init([hidden_size, input_size])
            self.beta  = zero_init([output_size, hidden_size + 1])
        # else:
        #     self.alpha = np.array([[ 1, 1, 2, 0,-1, 3, 2],
        #                            [ 1, 2, 3, 1, 0, 1, 1],
        #                            [ 1, 1, 3, 1, 2,-1, 2],
        #                            [ 1, 0, 1, 2, 0, 0, 3]])                                   
        #     self.beta  = np.array([[ 1, 1, 2, 0, 1],
        #                            [ 1, 1,-1, 3, 2],
        #                            [ 1, 3, 0,-1, 1]])

        # initialize parameters for adagrad
        self.epsilon = 0.00001
        self.grad_sum_alpha = np.zeros([hidden_size, input_size])
        self.grad_sum_beta = np.zeros([output_size, hidden_size + 1])

        # feel free to add additional attributes


def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.

    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)


def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data *with the bias folded in*
    :param nn: neural network class
    :return: output probability
    """
    nn.a = linear(X, nn.alpha)
    nn.z = sigmoid(nn.a)
    nn.b = linear(nn.z, nn.beta)
    y_hat = softmax(nn.b)
    return y_hat


def forward_clean(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data *with the bias folded in*
    :param nn: neural network class
    :return: output probability
    """
    a = linear(X, nn.alpha)
    z = sigmoid(a)
    b = linear(z, nn.beta)
    y_hat = softmax(b)
    return y_hat


def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data *with the bias folded in*
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_alpha: gradients for alpha
    d_beta: gradients for beta
    """
    g_b = d_cross_entropy_vec(y, y_hat)
    g_B, g_z = d_linear(nn.beta, nn.z, g_b)   
    g_a = np.multiply(g_z, d_sigmoid(nn)) 
    g_A, g_x = d_linear(nn.alpha, X, g_a)
    # print("shape of dl/db:\n", g_b.shape)
    # print("dl/db:\n", g_b)
    # print("shape of dl/dBeta:\n", g_B.shape)
    # print("dl/dBeta:\n", g_B)
    # print("shape of dl/dz:\n", g_z.shape)
    # print("dl/dz:\n", g_z)
    # print("shape of dl/da:\n", g_a.shape)
    # print("dl/da:\n", g_a)
    # print("shape of X:\n", X.shape)
    # print("X:\n", X)
    # print("shape of dl/dAlpha:\n", g_A.shape)
    # print("dl/dAlpha:\n", g_A)
    return g_A, g_B


def test(X, y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    predictions = []
    for i in range(len(X)):
        y_hat = forward_clean(X[i], nn)
        predictions.append(np.argmax(y_hat))
    return np.array(predictions)


def train(X_tr, y_tr, X_te, y_te, nn):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param X_te: train data
    :param y_te: train label
    :param nn: neural network class
    """
    train_cross_entropy_list = []
    test_cross_entropy_list = []
    for epoch in range(nn.n_epoch):
        # print("epoch: ", epoch)
        shuffle_X, shuffle_y = shuffle(X_tr, y_tr, epoch)
        for i in range(len(X_tr)):
            y_hat = forward(shuffle_X[i], nn)
            # print("\n\nsample ",i)
            # print("shape of a:\n", nn.a.shape)
            # print("a:\n", nn.a)
            # print("shape of z:\n", nn.z.shape)
            # print("z:\n", nn.z)
            # print("shape of b:\n", nn.b.shape)
            # print("b:\n", nn.b)
            # print("shape of y_hat:\n", y_hat.shape)
            # print("y_hat:\n", y_hat)
            # print("cross entropy:\n", cross_entropy(y_tr[i], y_hat))

            g_alpha, g_beta = backward(shuffle_X[i], shuffle_y[i], y_hat, nn)
            nn.grad_sum_alpha = nn.grad_sum_alpha + np.multiply(g_alpha, g_alpha)
            nn.grad_sum_beta  = nn.grad_sum_beta  + np.multiply(g_beta,  g_beta)
            # print("shape of grad_sum_alpha:\n", nn.grad_sum_alpha.shape)
            # print("grad_sum_alpha:\n", nn.grad_sum_alpha)
            # print("shape of grad_sum_beta:\n", nn.grad_sum_beta.shape)
            # print("grad_sum_beta:\n",  nn.grad_sum_beta)
            
            lr_array_alpha = nn.lr / np.sqrt(nn.grad_sum_alpha + nn.epsilon)
            lr_array_beta  = nn.lr / np.sqrt(nn.grad_sum_beta  + nn.epsilon)
            # print("shape of lr_array_alpha:\n", lr_array_alpha.shape)
            # print("lr_array_alpha:\n", lr_array_alpha)
            # print("shape of lr_array_beta:\n", lr_array_beta.shape)
            # print("lr_array_beta:\n", lr_array_beta)

            nn.alpha = nn.alpha - np.multiply(lr_array_alpha, g_alpha)
            nn.beta =  nn.beta  - np.multiply(lr_array_beta, g_beta)
            # print("shape of update_alpha:\n", nn.alpha.shape)
            # print("update_alpha:\n", nn.alpha)
            # print("shape of update_beta:\n", nn.beta.shape)
            # print("update_beta:\n",  nn.beta)
        
        train_cross_entropy = 0
        for i in range(len(X_tr)):
            y_hat_tr = forward_clean(X_tr[i], nn)
            # print(cross_entropy(y_tr[i], y_hat_tr))
            train_cross_entropy += cross_entropy(y_tr[i], y_hat_tr)
        train_cross_entropy /= len(y_tr)
        train_cross_entropy_list.append(train_cross_entropy)

        test_cross_entropy = 0
        for j in range(len(X_te)):
            y_hat_te = forward_clean(X_te[j], nn)
            # print(cross_entropy(y_te[j], y_hat_te))
            test_cross_entropy += cross_entropy(y_te[j], y_hat_te)
        test_cross_entropy /= len(y_te)
        test_cross_entropy_list.append(test_cross_entropy)
    
    return train_cross_entropy_list, test_cross_entropy_list

def compute_error(y_pred, y):
    error = 0
    for i in range(len(y)):
        if y_pred[i] != y[i]: error += 1
    return error/len(y)

if __name__ == "__main__":

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate

    # initialize training / test data and labels
    X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics, n_epochs, n_hid, init_flag, lr = args2data(args)

    # Build model
    # print(len(X_tr[0]))
    my_nn = NN(lr, n_epochs, args.init_flag, len(X_tr[0]), n_hid, 10)

    # train model
    tr_CE, te_CE = train(X_tr, y_tr, X_te, y_te, my_nn)
    # print(tr_CE)
    # print(te_CE)

    # test model and get predicted labels and errors
    train_predictions = test(X_tr, y_tr, my_nn)
    test_predictions  = test(X_te, y_te, my_nn)
    np.savetxt(out_tr, train_predictions.astype(int), 
                fmt = '%i', delimiter = "\n") #4
    np.savetxt(out_te, test_predictions.astype(int), 
                fmt = '%i', delimiter = "\n") #5

    error_train = compute_error(train_predictions, y_tr)
    error_test  = compute_error(test_predictions,  y_te)

    # write predicted label and error into file
    with open(out_metrics, 'w') as f: #6
        for i in range(n_epochs):
            f.write("epoch=" + str(i+1) + " crossentropy(train): " + str(tr_CE[i]) + '\n')
            f.write("epoch=" + str(i+1) + " crossentropy(validation): " + str(te_CE[i]) + '\n')
        f.write("error(train): " + str(error_train) + '\n')
        f.write("error(test): " + str(error_test) + '\n')
