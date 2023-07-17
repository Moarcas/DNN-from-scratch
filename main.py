import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    np.random.seed(1)
    grads = {}
    costs = []
    parameters = initialize_parameters(layers_dims[0], layers_dims[1], layers_dims[2])

    for iteration in range(num_iterations):
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]

        # Forward step
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        # Compute cost
        cost = compute_cost(A2, Y)

        # Backward step
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        grads["dA1"], grads["dW2"], grads["db2"] = linear_activation_backward(dA2, cache2, "sigmoid")
        grads["dA0"], grads["dW1"], grads["db1"] = linear_activation_backward(grads["dA1"], cache1, "relu")

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and iteration % 100 == 0:
            print("Cost after iteration {}: {}".format(iteration, np.squeeze(cost)))
            costs.append(cost)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    np.random.seed(1)
    
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255
    test_x = test_x_flatten/255
    
    n_x = 12288
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)

    parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)
    