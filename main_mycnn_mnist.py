from dl_cnn import NN
import utility as util
from utility import HistoryRecord


from sklearn.model_selection import train_test_split
import numpy as np

## Hyper parameters
BATCH_SIZE = 32
N_EPOCH = 10
LEARNING_RATE = 0.0001
MOMENTUM = 0.9


## load mnist dataset
X_train, X_test, y_train, y_test = util.load_mnist()

## split into training and validating sets
# X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2)


## set validation data
# validate_data = {'validating':(X_validate, y_validate), 'testing':(X_test, y_test)}
validate_data = {'testing':(X_test, y_test)}

history = HistoryRecord(validate_data, 100)

## build CNN
dnn = NN(lr=LEARNING_RATE, momentum=MOMENTUM, callback=history)

dnn.set_input_data(X_train)
dnn.set_target(y_train)


dnn.add_conv2d_layer(8, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_conv2d_layer(16, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_conv2d_layer(32, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=2)
dnn.add_flatten_layer()
dnn.add_FC_layer(n_hidden=10, activation=False)
dnn.add_stable_softmax_cross_entropy_layer()
# dnn.add_weight_decay(penalty_weight=10.0)

dnn.train(n_epoch=N_EPOCH, mini_batch_size=BATCH_SIZE)

tracking_number = util.get_random_string(3)
util.save_obj('cnn_epoch{}_batch{}_{}.pickle'.format(N_EPOCH, BATCH_SIZE, tracking_number), dnn)
util.save_obj('history_{}.pickle'.format(tracking_number), history)
