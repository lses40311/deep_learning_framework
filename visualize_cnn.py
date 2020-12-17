'''
usage: python visualize_cnn.py [model_path] [dataset]
dataset: 'mnist' or 'cifar'
'''

import matplotlib.pyplot as plt
import sys
import numpy as np
import utility as util
from utility import HistoryRecord

cmap = 'Greys'

## load the model
if len(sys.argv) == 3:
    print('Loading model {} for {}'.format(sys.argv[1], sys.argv[2]))
    dnn = util.load_obj(sys.argv[1])
    dataset = sys.argv[2]
elif len(sys.argv) == 2:
    print('Loading model dl_class.pickle')
    dnn = util.load_obj('dl_class.pickle')
    dataset = sys.argv[2]
else:
    dnn = util.load_obj('dl_class.pickle')
    dataset = 'mnist'

if dataset == 'mnist':
    ## load mnist dataset
    X_train, X_test, y_train, y_test = util.load_mnist()
    ch = 0
elif dataset == 'cifar':
    ch = [0,1,2]
    path_to_training_data = 'dataset/cifar-10-batches-py/data_batch_1'
    path_to_testing_data = 'dataset/cifar-10-batches-py/test_batch'
    X_train, X_test, y_train, y_test = util.data_loader_cifar(path_to_training_data, path_to_testing_data)


print(dnn.G.print_edges())

## 1-1
## task 1-1 Accuracies
histogram_range = (-1,1)
figsize=(10,5)
dpi = 200

## plot accuracies
train_acc = [value[0] for key, value in dnn.callback.accuracies.items()]
test_acc = [value[1] for key, value in dnn.callback.accuracies.items()]
plt.figure(figsize=figsize)
plt.plot(train_acc)
plt.plot(test_acc)

plt.ylabel('Accuracy')
plt.xlabel('Minibatch #')
plt.legend(['Train', 'Test'], loc='upper left')
plt.title('Training Accuracy')

plt.savefig('1-1_accuracy.jpg', format='jpg', dpi=dpi)

## plot learnign curve
## loss
train_losses = [value for key, value in dnn.callback.losses.items()]
plt.figure(figsize=figsize)
plt.plot(train_losses)

plt.ylabel('Loss')
plt.xlabel('Minibatch #')
plt.legend(['Training'], loc='upper right')
plt.title('Learning Curve')

plt.savefig('1-1_loss.jpg', format='jpg', dpi=dpi)

## histogram of weights
if dataset == 'mnist':
    offset = 0
    ch = 0
    var_name = {'conv1':'K_0', 'conv2':'K_1', 'dense':'W_5', 'output':'a_5'} ## mnist
    label_names = {0:'0', 1:'1', 2:'2', 3:'3',
        4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'}
elif dataset == 'cifar':
    offset = 0.47
    ch = [0,1,2]
    var_name = {'conv1':'K_0', 'conv2':'K_2', 'dense':'W_7', 'output':'a_8'} ## cifar 32-64-128
    label_names = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat',
        4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
else:
    assert 1 == 0

weights = dnn.G.vertex_container['K_0'].value.flatten()
plt.figure(figsize=figsize)
plt.hist(weights, bins=100, range=histogram_range)
plt.title('Weights of Conv 1')
plt.savefig('1-1_weights_L0.jpg', format='jpg', dpi=dpi)

## histogram of weights
weights = dnn.G.vertex_container[var_name['conv2']].value.flatten()
plt.figure(figsize=figsize)
plt.hist(weights, bins=100, range=histogram_range)
plt.title('Weights of Conv 2')
plt.savefig('1-1_weights_L1.jpg', format='jpg', dpi=dpi)

## histogram of weights
weights = dnn.G.vertex_container[var_name['dense']].value.flatten()
plt.figure(figsize=figsize)
plt.hist(weights, bins=100, range=histogram_range)
plt.title('Weights of Dense 1')
plt.savefig('1-1_weights_W3.jpg', format='jpg', dpi=dpi)

## histogram of weights
weights = dnn.G.vertex_container[var_name['output']].value.flatten()
plt.figure(figsize=figsize)
plt.hist(weights, bins=100, range=histogram_range)
plt.title('Weights of Output')
plt.savefig('1-1_weights_a3.jpg', format='jpg', dpi=dpi)


## 1-3
## plot feature maps
if dataset == 'mnist':
    var_name = {'conv1':'a_0', 'conv2':'a_1'} ## MNIST
elif dataset == 'cifar':
    var_name = {'conv1':'a_0', 'conv2':'a_2'} ## MNIST
else:
    assert 1 == 0

im_idx = 5
out = dnn.G.vertex_container['x'].value.transpose((0,2,3,1))
if dataset == 'mnist':
    fm = out[im_idx,:,:,0]
elif dataset == 'cifar':
    fm = out[im_idx,:,:,:] + offset
plt.imsave('1-3_origin.jpg', fm,  format='jpg', dpi=dpi)

## plot all feature maps - L0 kernels 'a_0'
out = dnn.G.vertex_container[var_name['conv1']].value
n_kernels = out.shape[1]
n_row = int(np.ceil(np.sqrt(n_kernels)))
fig, axes = plt.subplots(n_row , n_row, figsize=(20,20))

for i in range(n_kernels):
    ax = axes[int(i/n_row),int(i%n_row)]
    ax.imshow(out[im_idx][i], cmap=cmap)
    ax.title.set_text("Feature map {}".format(i))

fig.savefig('1-3_featuremap_L0.jpg', format='jpg', dpi=dpi)


## plot all feature maps - L1 kernels 'a_1'
out = dnn.G.vertex_container[var_name['conv2']].value
n_kernels = out.shape[1]
n_row = int(np.ceil(np.sqrt(n_kernels)))
fig, axes = plt.subplots(n_row , n_row, figsize=(20,20))

for i in range(n_kernels):
    ax = axes[int(i/n_row),int(i%n_row)]
    ax.imshow(out[im_idx][i], cmap=cmap)
    ax.title.set_text("Feature map {}".format(i))

fig.savefig('1-3_featuremap_L1.jpg', format='jpg', dpi=dpi)


## 1-2
## find misclassified example
ix = 68
pred = dnn.predict(X_test[np.newaxis, ix])

wrong_ix = 16
pred_wrong = dnn.predict(X_test[np.newaxis, wrong_ix])

fig, axes = plt.subplots(1, 2)

if dataset == 'mnist':
    axes[0].imshow(X_test[ix,0,:,:])
    axes[1].imshow(X_test[wrong_ix,0,:,:])
elif dataset == 'cifar':
    axes[0].imshow(X_test[ix,:,:,:].transpose((1,2,0))+ offset)
    axes[1].imshow(X_test[wrong_ix,:,:,:].transpose((1,2,0))+ offset)

axes[0].title.set_text("Label: {}, Predict: {}".format(label_names[np.argmax(y_test[ix])], label_names[np.argmax(pred)]))
axes[1].title.set_text("Label: {}, Predict: {}".format(label_names[np.argmax(y_test[wrong_ix])], label_names[np.argmax(pred_wrong)]))

fig.savefig('1-2.jpg', format='jpg', dpi=dpi)
