#!/usr/bin/python

#########################################
# module: cs5600_6600_f24_hw02.py
# description: some starter code
# bugs to vladimir kulyukin in canvas.
#########################################

import numpy as np
import pickle
from cs5600_6600_f24_hw02_data import *
# from cs5600_6600_f24_hw02_uts import *

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
    
# save() function to save the trained network to a file
def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def build_nn_wmats(mat_dims):
    if len(mat_dims) < 2:
        raise Exception("Must have at least two dimensions")
    else:
        return tuple([np.random.randn(mat_dims[i], mat_dims[i+1]) for i in range (len(mat_dims) - 1)])

def build_231_nn():
    return build_nn_wmats((2, 3, 1))

def build_2331_nn():
    return build_nn_wmats((2, 3, 3, 1))

def build_221_nn():
    return build_nn_wmats((2, 2, 1))

def build_838_nn():
    return build_nn_wmats((8, 3, 8))

def build_949_nn():
    return build_nn_wmats((9, 4, 9))

def build_4221_nn():
    return build_nn_wmats((4, 2, 2, 1))

def build_421_nn():
    return build_nn_wmats((4, 2, 1))

def build_121_nn():
    return build_nn_wmats((1, 2, 1))

def build_1221_nn():
    return build_nn_wmats((1, 2, 2, 1))

# 3 layer build methods
layer_3_methods =  [build_121_nn, build_221_nn, build_231_nn, build_838_nn, build_949_nn, build_421_nn]

## Training 3-layer neural net.
## X is the matrix of inputs
## y is the matrix of ground truths.
## build is a nn builder function.
def train_3_layer_nn(numIters, X, y, build):
    # 1) constructs appropriate number of w mats
    W1, W2 = build()

    # 2) Trains the ANN for the specified number of iterations on x and y using FF abd BP 
    for _ in range(numIters):
        # FF
        # Layer 1 to 2
        Z2 = np.dot(X, W1)
        a2 = sigmoid(Z2)

        # Layer 2 to output
        Z3 = np.dot(a2, W2)
        y_hat = sigmoid(Z3)

        # BP
        # Output Layer
        y_hat_error = y - y_hat
        y_hat_delta = y_hat_error * sigmoid(y_hat, deriv=True)
        
        # layer 2
        a2_error = y_hat_delta.dot(W2.T)
        a2_delta = a2_error * sigmoid(a2, deriv=True)
        
        # layer 3
        W2 += a2.T.dot(y_hat_delta)
        W1 += X.T.dot(a2_delta) 
    
    return (W1, W2)

def train_4_layer_nn(numIters, X, y, build):
    # 1) constructs appropriate number of w mats
    W1, W2, W3 = build()

    # 2) Trains the ANN for the specified number of iterations on x and y using FF abd BP 
    for _ in range(numIters):
        # FF
        # Layer 1 to 2
        Z2 = np.dot(X, W1)
        a2 = sigmoid(Z2)

        # Layer 2 to 3
        Z3 = np.dot(a2, W2)
        a3 = sigmoid(Z3)

        # layer 3 to Output
        Z4 = np.dot(a3, W3)
        y_hat = sigmoid(Z4)

        # BP
        # Output layer
        y_hat_error = y - y_hat
        y_hat_delta = y_hat_error * sigmoid(y_hat, deriv=True)

        # Layer 3
        a3_error = y_hat_delta.dot(W3.T)
        a3_delta = a3_error * sigmoid(a3, deriv=True)

        # Layer 2
        a2_error = a3_delta.dot(W2.T)
        a2_delta = a2_error * sigmoid(a2, deriv=True)

        # Update W
        W3 += a3.T.dot(y_hat_delta)
        W2 += a2.T.dot(a3_delta)
        W1 += X.T.dot(a2_delta) 
    
    return (W1, W2, W3)

def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    W1, W2 = wmats

    # ff
    # layer 1 to 2
    Z2 = np.dot(x, W1)
    a2 = sigmoid(Z2)

    # Layer 2 to output
    Z3 = np.dot(a2, W2)
    y_hat = sigmoid(Z3)

    # Thresholding
    if thresh_flag:
        return (y_hat >= thresh).astype(int)

    return y_hat

def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    W1, W2, W3 = wmats

    # ff
    # layer 1 to 2
    Z2 = np.dot(x, W1)
    a2 = sigmoid(Z2)

    # Layer 2 to 3
    Z3 = np.dot(a2, W2)
    a3 = sigmoid(Z3)

    # Layer 3 to output
    Z4 = np.dot(a3, W3)
    y_hat = sigmoid(Z4)

    # Thresholding
    if thresh_flag:
        return (y_hat >= thresh).astype(int)

    return y_hat

def save_successful_nn(file_name, build_method, x, y, thresh = 0.4, fail_limit = 10):
    fail_count = 0
    successful = False

    def validate(input_weights, fit_function, thresh):
        for i in range(len(x)):
            print('{}, {} --> {}'.format(x[i], fit_function(x[i], input_weights), y[i]))
            assert (fit_function(x[i], input_weights, thresh=thresh, thresh_flag=True) == y[i]).all()
    while (not successful and fail_count < fail_limit):
        try:
            print(f"\n\nAttempt {fail_count + 1}")
            np.random.seed(fail_count)
            if build_method in layer_3_methods:
                ann_obj = train_3_layer_nn(500, x, y, build_method)
                fit_function = fit_3_layer_nn
            else:
                ann_obj = train_4_layer_nn(500, x, y, build_method)
                fit_function = fit_4_layer_nn
            validate(ann_obj, fit_function, thresh)
            save(ann_obj, file_name=file_name)
            successful = True
        except AssertionError:
            print("Assertion Error reached!")
            fail_count += 1
        except ValueError as e:
            print(f"Issue with weights: {e}")
            break
    if successful:
        print(f"saved {file_name} successfully after {fail_count} failures.")
    else:
        print(f"{file_name} training ended in failure!")

### Save pickled networks

# 1. and_3_layer_ann.pck;
# save_successful_nn("and_3_layer_ann.pck", build_231_nn, X1, y_and)

# # 2. and_4_layer_ann.pck;
# save_successful_nn("and_4_layer_ann.pck", build_2331_nn, X1, y_and)

# # 3. or_3_layer_ann.pck;
# save_successful_nn("or_3_layer_ann.pck", build_231_nn, X1, y_or)

# # 4. or_4_layer_ann.pck;
# save_successful_nn("or_4_layer_ann.pck", build_2331_nn, X1, y_or)

# # 5. not_3_layer_ann.pck;
# save_successful_nn("not_3_layer_ann.pck", build_121_nn, X2, y_not)

# # 6. not_4_layer_ann.pck;
# save_successful_nn("not_4_layer_ann.pck", build_1221_nn, X2, y_not)

# # 7. xor_3_layer_ann.pck;
# save_successful_nn("xor_3_layer_ann.pck", build_231_nn, X1, y_xor)

# # 8. xor_4_layer_ann.pck
# save_successful_nn("xor_4_layer_ann.pck", build_421_nn, X3, bool_exp, thresh=0.3)

# 9. bool_3_layer_ann.pck;
save_successful_nn("bool_3_layer_ann.pck", build_421_nn, X3, bool_exp, thresh=0.3)

# 9. bool_3_layer_ann.pck;
save_successful_nn("bool_4_layer_ann.pck", build_4221_nn, X3, bool_exp, thresh=0.3)