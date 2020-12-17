import pickle
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import string
import random


class HistoryRecord:
    def __init__(self, validation_data_dict, batch_size):
        ## {'test': (X:np.array, y:np.array)}
        self.validation_data = validation_data_dict
        self.accuracies = {}
        self.losses = {}
        self.batch_size = batch_size

    def get_testing_data(self):
        ## returns #(batchsize) of datapoints from every validtiona dastasets
        # n_dataset = [len(value[0]) for key, value in self.validation_data.items()]
        if self.batch_size:
            random_indeces =  [np.arange(len(value[0])) for key, value in self.validation_data.items()]
            for i in range(len(random_indeces)):
                np.random.shuffle(random_indeces[i])
                random_indeces[i] = random_indeces[i][:self.batch_size]
            X_validate_lst = [value[0][random_indeces[ix]] for ix, (key, value) in enumerate(self.validation_data.items())]
            y_validate_lst = [value[1][random_indeces[ix]] for ix, (key, value) in enumerate(self.validation_data.items())]
            return X_validate_lst, y_validate_lst
        else:
            return [value[0] for key, value in self.validation_data.items()], [value[1] for key, value in self.validation_data.items()]

    def record_score(self, epoch, batch, scores, losses):
        record_name = str(epoch) +'_'+ str(batch)
        self.accuracies[record_name] = scores
        self.losses[record_name] = losses

def save_obj(name, obj):
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as handle:
        return pickle.load(handle)

def get_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def load_mnist():
    ## Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    ## one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    X_train = np.transpose(X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)), (0, 3, 1, 2))
    # X_train = (X_train / 255)

    X_test = np.transpose(X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)), (0, 3, 1, 2))
    # X_test = (X_test / 255)

    X_train, X_test = zero_center(X_train, X_test)

    return X_train, X_test, y_train, y_test


def data_loader_cifar(path_to_training_data, path_to_testing_data):
    with open(path_to_training_data, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    X_train = batch['data'].reshape((len(batch['data']), 3, 32, 32)).astype('float32')
    y_train = to_categorical(batch['labels'])

    with open(path_to_testing_data, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    X_test = batch['data'].reshape((len(batch['data']), 3, 32, 32)).astype('float32')
    y_test = to_categorical(batch['labels'])

    X_train, X_test = zero_center(X_train, X_test)

    return X_train, X_test, y_train, y_test

def show_cifar_im(x):
    plt.imshow(x.transpose((1,2,0)))

def zero_center(X_train, X_test):
    mean = np.mean(X_train, axis=(0,1,2,3))
    X_train = (X_train-mean)/255.0
    X_test = (X_test-mean)/255.0
    return X_train, X_test

def z_score(X_train, X_test):
    #z-score
    mean = np.mean(X_train, axis=(0,1,2,3))
    std = np.std(X_train, axis=(0,1,2,3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test
