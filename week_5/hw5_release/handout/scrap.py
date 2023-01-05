import numpy as np
import sys
import matplotlib.pyplot as plt

def sigmoid(a):
    """
    Implementation of the sigmoid function.

    Parameters:
        a (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(a)
    return e / (1 + e)

shape = [10,5]
np.random.seed(np.prod(shape))

# result = np.random.uniform(low = -0.1, high = 0.1, size = shape)
result = np.zeros(shape)
bias_fold = np.ones(len(result), dtype = float)
result = sigmoid(np.vstack((bias_fold, result.transpose()[1:])).transpose())

result = np.sqrt(result)

a = np.array([1,1,1,5,2])

print(np.argmax(a))