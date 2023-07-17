import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = True):
    np.random.seed(1)

    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for iteration in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and iteration % 100 == 0:
            print ("Cost after iteration %i: %f" %(iteration, cost))
            costs.append(cost)
            
    
    plt.plot(np.squeeze(cost))
    plt.xlabel("iterations (per hundreds)")
    plt.ylabel("cost")
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
    
    layers_dims = [12288, 20, 7, 5, 1]

    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)
    