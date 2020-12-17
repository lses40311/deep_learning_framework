from dl_cnn import NN
import utility as util
from utility import HistoryRecord


## loading batch 1
path_to_training_data = 'dataset/cifar-10-batches-py/data_batch_1'
path_to_testing_data = 'dataset/cifar-10-batches-py/test_batch'
X_train, X_test, y_train, y_test = util.data_loader_cifar(path_to_training_data, path_to_testing_data)

## Hyper parameters
BATCH_SIZE = 32
N_EPOCH = 10
# LEARNING_RATE = 0.0000001
LEARNING_RATE = 0.0001
MOMENTUM = 0.8


## set validation data
validate_data = {'testing':(X_test, y_test)}
history = HistoryRecord(validate_data, 120)

## build CNN
dnn = NN(lr=LEARNING_RATE, momentum=MOMENTUM, callback=history, clipping=2000, weight_decay=False)

dnn.set_input_data(X_train)
dnn.set_target(y_train)


dnn.add_conv2d_layer(32, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=2)

dnn.add_conv2d_layer(64, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=2)

dnn.add_conv2d_layer(128, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=2)

dnn.add_conv2d_layer(256, kernel_size=(3,3), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=2)

dnn.add_flatten_layer()
dnn.add_FC_layer(n_hidden=128, activation='relu')
dnn.add_FC_layer(n_hidden=64, activation='relu')
dnn.add_FC_layer(n_hidden=32, activation='relu')
dnn.add_FC_layer(n_hidden=10, activation=False)
dnn.add_stable_softmax_cross_entropy_layer()
# dnn.add_weight_decay(penalty_weight=2.5)

dnn.train(n_epoch=N_EPOCH, mini_batch_size=BATCH_SIZE)

tracking_number = util.get_random_string(3)
util.save_obj('tmp/cnn_epoch{}_batch{}_{}.pickle'.format(N_EPOCH, BATCH_SIZE, tracking_number), dnn)
util.save_obj('tmp/history_{}.pickle'.format(tracking_number), history)
