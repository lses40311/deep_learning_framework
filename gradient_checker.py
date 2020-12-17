from dl_cnn import NN
import utility as util

import itertools
import numpy as np
from random import sample

'''
Testing the parameter in the first layer == testing most of the parameters in
the following layers since the gradients backproped through other parameters first.
'''
## K_0: the filter in the layer 0
test_variable = 'K_0'

## how many weights you want to test
n_sampled_parameter = 50

## epsilon
epsilon = 0.0001

## the threshold of passing the test
threshold = 0.01


# loading batch 1
path_to_training_data = 'dataset/cifar-10-batches-py/data_batch_1'
path_to_testing_data = 'dataset/cifar-10-batches-py/test_batch'
X_train, X_test, y_train, y_test = util.data_loader_cifar(path_to_training_data, path_to_testing_data)


## Use 50 data points
X_train = X_train[:50]
y_train = y_train[:50]

## Build NN
dnn = NN(lr=0.00001)

## set input manually
dnn.set_input_data(X_train)
dnn.set_target(y_train)
dnn.G.put_input_data(X_train, y_train)

## add layers
dnn.add_conv2d_layer(16, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_conv2d_layer(16, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=2)

dnn.add_conv2d_layer(32, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=2)

dnn.add_conv2d_layer(64, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=2)

dnn.add_flatten_layer()
dnn.add_FC_layer(n_hidden=64, activation='relu')
dnn.add_FC_layer(n_hidden=10, activation=False)

dnn.add_stable_softmax_cross_entropy_layer()
## add layers



dnn.grad_table = {dnn.loss_variable_name: np.array(1.0)}
dnn.G.clear_ops()
dnn.G.compute(dnn.loss_variable_name)
print('loss: {}'.format(dnn.G.vertex_container[dnn.loss_variable_name].value))
grad = dnn.build_grad(test_variable)


## sample the weighs to be tested
test_parameter = dnn.G.vertex_container[test_variable].value
variable_dimension = test_parameter.ndim
n_dims = []
for axis in range(variable_dimension):
    n_dims.append(list(range(test_parameter.shape[axis])))
all_indices = list(itertools.product(*n_dims))

if n_sampled_parameter > len(all_indices):
    n_sampled_parameter = len(all_indices)
sample_indices = sample(all_indices, n_sampled_parameter)

n_pass = 0
for index in sample_indices:
    param_plus = test_parameter.copy()
    param_plus[index] += epsilon
    dnn.G.vertex_container[test_variable].value = param_plus
    dnn.G.clear_ops()
    dnn.G.compute(dnn.loss_variable_name)
    loss_plus = dnn.G.vertex_container[dnn.loss_variable_name].value

    param_minus = test_parameter.copy()
    param_minus[index] -= epsilon
    dnn.G.vertex_container[test_variable].value = param_minus
    dnn.G.clear_ops()
    dnn.G.compute(dnn.loss_variable_name)
    loss_minus = dnn.G.vertex_container[dnn.loss_variable_name].value

    grad_analytic = (loss_plus - loss_minus)/(2*epsilon)
    diff = abs(grad_analytic - grad[index])/max(abs(grad_analytic), abs(grad[index]))
    if diff < threshold:
        print('{} {:13} gradient match diff={:2f}'.format(test_variable, str(index), diff))
        n_pass += 1
    else:
        print('{} {:13} gradient Not match, diff={:2f}'.format(test_variable, str(index), diff))

print('The rate of passing the test is {}.'.format(float(n_pass)/float(n_sampled_parameter)))
