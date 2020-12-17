# Deep Learning Framework

## Overview

This project implemented a neural network framework `dl_cnn.py` following to the algorithms introduced in the [Deep Learning textbook](https://www.deeplearningbook.org/). This framework is consists of three main classes: `class NN`, `class ComputationalGraph`, and `class Variable`.

- `class NN` is responsible for building and training the neural network.
    - `set_input_data(X)` and `set_target(y)` are functions for setting the input data and the target labels.
    - Fully-connected layers can be added to the network by using function `add_FC_layer(n_hidden, activation)`.
    - The output layer for calculating the mean squared error can be added to the network by using `add_mse_layer()`.
    - `back_propagation()` forward passes the data and back propagates the result for obtaining the gradient, and performs a step of gradient decent on the parameters. The back propagation function heavily relies on calling the `build_grad(V)` function, which was implemented by following the algorithm 6.5 and 6.6.
- `class Variable` is a data structure for storing the variable in the computational graph.
- `class ComputationalGraph` handles the construction of the computational graph, such as inserting and deleting variable vertex. This class also contains three abstract functions described in the textbook p.p. 215, namely `get_operation(V)`, `get_consumers(V)`, and `get_inputs(V)`.

Operation classes `class op_*` are the operations in the computational graph. Each operation has two member functions for forward pass `f()` and back-propagation `bprop()`, separately. We have implemented the basic operations for constructing convolutional neural networks:

- `class op_conv2d` was implemented for performing convolution and backpropagation. The convolution was performed by following the [lecture notes of CS231n](https://cs231n.github.io/convolutional-networks/). In this framework, we implemented an im2row() function for fast convolution. The backpropagation followed the instruction in [this post](https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer).
- `class op_max_pool_2d` was implemented for max-pooling. The back propagation was implemented by following [this post](http://op_max_pool_2d/).
- `class op_flatten` reshapes the tensor into a matrix for fully connected layers.
- `class op_stable_softmax_cross_entropy` was implemented for preventing the system from overflow and underflow. This problem is described at the end of this report.

We have also implemented some auxiliary functions for testing the deep learning framework:

- `utility.py` contains utilities such as loading dataset, saving objects… etc.
- `main_mycnn_mnist.py` builds and trains the CNN model, and then saves it as a .pickle file for the MNIST dataset.
- `main_mycnn_cifar.py` builds and trains the CNN model, and then saves it as a .pickle file for the CIFAR-10 dataset.
- `visualize_cnn.py` plots the figures for tasks 1 and 2 with the saved models trained on the tasks. Usage: `python visualize_cnn.py [model] [dataset]`. You can download one of our pretrained models [here](https://drive.google.com/file/d/1I8GeFWmTfPd1qKEVG6eE1sXXIkuvYQKC/view?usp=sharing) and run the visualization to obtain the figures in this report.
- `gradient_checker.py` checks the gradients of a parameter in the network by comparing the numerical gradient and the analytic gradient.

In the following, we will demonstrate the usage of this deep learning framework on various types of task.

## Task 1: Regression using fully connected networks

### Minimizing the sum-of-squares error

The goal of this part is to perform a regression task for predicting the energy consumption based on the characteristics of the house. The project includes a jupyter notebook which demonstrates the building and training of the neural network. Simply call the `set_input()` function to add an input layer to the network and add fully-connected layers with specified number of hidden nodes by calling `add_FC_layer()`, which adds a layer for processing the data with equation $\textbf{A}_i = \sigma(\textbf{A}_{i-1}\textbf{W}_i+\textbf{b}_i)$, where $\textbf{A}_i$is the output of layer $i$, $\textbf{W}_i$is the weights of layer $i$, and $\textbf{b}_i$is the bias of layer $i$.  The last fully-connected layer has only one node to perform regression without an activation function. And finally add a `MSE layer` for calculating the cost. The code for building the network is shown below:

```python
from dl_cnn import ComputationalGraph, Variable, NN
dnn = NN(lr=0.000001

## building the neural network
dnn.set_input_data(X_train)
dnn.set_target(y_train)
dnn.add_FC_layer(n_hidden=5, activation='relu')
dnn.add_FC_layer(n_hidden=6, activation='relu')
dnn.add_FC_layer(n_hidden=7, activation='relu')
dnn.add_FC_layer(n_hidden=5, activation='relu')
dnn.add_FC_layer(n_hidden=4, activation='relu')
dnn.add_FC_layer(n_hidden=1, activation=False)

## add a loss function
dnn.add_mse_layer()
```

The function `data_loader_regression()` was implemented for preprocessing (one-hot encoding) and normalizing ($x' = \frac{x - \mu}{max(x) - min(x)}$) the features. 

### Prediction results

[Untitled](https://www.notion.so/763dc61248ae45238c8bda24c853db31)

The figure below shows the learning curve of the neural network. The error fluctuated a little at the beginning then went down fast, and eventually saturated.

![Deep%20Learning%20Framework%200c344b6483bb4861b14159c99e7cc8d5/learning_curve.png](Deep%20Learning%20Framework%200c344b6483bb4861b14159c99e7cc8d5/learning_curve.png)

The two figures below show the prediction results on the training data and the testing data.

![Deep%20Learning%20Framework%200c344b6483bb4861b14159c99e7cc8d5/training_prediction.png](Deep%20Learning%20Framework%200c344b6483bb4861b14159c99e7cc8d5/training_prediction.png)

![Deep%20Learning%20Framework%200c344b6483bb4861b14159c99e7cc8d5/testing_prediction.png](Deep%20Learning%20Framework%200c344b6483bb4861b14159c99e7cc8d5/testing_prediction.png)

## Task 2: Classification using fully connected networks

### Minimizing the cross-entropy loss

The dataset was preprocessed by the function data_loader_regression(). The features were normalized by the same method in the regression task and the categorical target labels were encoded with one-hot encoding.

For the classification task, I implemented two more operations `op_softmax` and `op_cross_entropy` in the framework. The input data and target labels were set by the functions `set_input()`  and `set_target()`. The procedure of adding fully-connected layers were the same as the regression task. The output layer was set to a fully-connected layer of two hidden nodes without activation function, and a cross-entropy node was appended for calculating the loss.  

```python
dnn = NN(lr=0.0001)
dnn.set_input_data(X_train)
dnn.set_target(y_train)
dnn.add_FC_layer(n_hidden=5)
dnn.add_FC_layer(n_hidden=8)
dnn.add_FC_layer(n_hidden=10)
dnn.add_FC_layer(n_hidden=6)
dnn.add_FC_layer(n_hidden=2, activation=False)
dnn.add_softmax_layer()
dnn.add_cross_entropy_layer()

errs = dnn.train(n_epoch=5000, mini_batch_size=64)
```

[Untitled](https://www.notion.so/ed9c5b3b7c5d4cf1a31c87bc5b205fa3)

The loss function (cross-entropy) is more or less correlated to the number of input data points, and the size of mini-batch might vary, so I got a fluctuating learning curve. To present the results clearly, I added the accuracies ($acc = \frac{correctly~ classified~ points}{total~ data~ points}$). The results showed that the training error is lower than the testing error. 

![Deep%20Learning%20Framework%200c344b6483bb4861b14159c99e7cc8d5/learning_curve_classification.png](Deep%20Learning%20Framework%200c344b6483bb4861b14159c99e7cc8d5/learning_curve_classification.png)

---

## Task 3: MNIST Classification

In the following, we present the results of training a CNN by the self-implemented package. The code is included in the file `main_mycnn_mnist.py`, which can be executed by command python `main_mycnn_mnist.py`. The CNN was a 8 Conv-16 Conv-32 Conv-Max Pool-10 Dense model. The training loss and accuracies were recorded every 5 mini batches. Minibatch size = 32, learning rate = 0.0001, and momentum = 0.9. The loss converged to around 5.0 with fluctuation (without normalization, should be divided by the size of minibatch), and the accuracy was around 95%. The code of constructing the network is shown below.

```python
# build CNN
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
```

[https://lh3.googleusercontent.com/GH1ItN8aS1v1AEMCo9ghZRivwAmF9FCO_MAvKGWhJTAMNP26uklPPl2Q4sEJb1eCGHKFnImzmk0zB1h9OnyF9iRPoHMtU4R6HdzwLTwCox2DLhiL5gL8zgpRksQ_ni3pBzmX-Oal](https://lh3.googleusercontent.com/GH1ItN8aS1v1AEMCo9ghZRivwAmF9FCO_MAvKGWhJTAMNP26uklPPl2Q4sEJb1eCGHKFnImzmk0zB1h9OnyF9iRPoHMtU4R6HdzwLTwCox2DLhiL5gL8zgpRksQ_ni3pBzmX-Oal)

[https://lh3.googleusercontent.com/X_9r75QhzWCPul5fJCtiV31L4c29wfIaWdWrnTpZLsPlLVuzEUn3QbacobYsnoeXzohQ7s2etAw39eOdq56H0Lxe4P3jt5mYulYwnLua3ymiPF81f-G_XapCRsh_CpeJG1S-nhvR](https://lh3.googleusercontent.com/X_9r75QhzWCPul5fJCtiV31L4c29wfIaWdWrnTpZLsPlLVuzEUn3QbacobYsnoeXzohQ7s2etAw39eOdq56H0Lxe4P3jt5mYulYwnLua3ymiPF81f-G_XapCRsh_CpeJG1S-nhvR)

### Prediction results

The figure below shows a misclassified image and an image of the class. The figure on the bottom left is a handwritten 2, but the orientation was a little off and classified as a 7. The figure on the bottom right is a correctly classified 7. We can observe their similarity.

[https://lh3.googleusercontent.com/fmdYLw5ujOivjVXszqlD-bKCITiIOFstphVJrl_7JxLFqyuXly0oo0gbs5vh9gijNr8eLfrqVm4id0ilsqo5DeemY3G4d8L8ljWlpz0H6khseCDK6iHDCmVY0yI0CcR0XuTtP1aH](https://lh3.googleusercontent.com/fmdYLw5ujOivjVXszqlD-bKCITiIOFstphVJrl_7JxLFqyuXly0oo0gbs5vh9gijNr8eLfrqVm4id0ilsqo5DeemY3G4d8L8ljWlpz0H6khseCDK6iHDCmVY0yI0CcR0XuTtP1aH)

### Feature map visualization

We selected an image of 2 and plotted all the feature maps on the first and second layers, which were set to have 8 and 16 kernels in this case. The original image is shown as follow:

[https://lh5.googleusercontent.com/BLNK6bqBGipRsaatej0haYU7URNzKTGh11n1P55bwBg8XFt78ayK2JMjHJBeYW_Ac39o2mhaHr5iObcPP_8vjLZF9yeL2yzEV_QsCzVRy8u69lRi4L9SQLLm7KN1H88ZSYGnYRv2](https://lh5.googleusercontent.com/BLNK6bqBGipRsaatej0haYU7URNzKTGh11n1P55bwBg8XFt78ayK2JMjHJBeYW_Ac39o2mhaHr5iObcPP_8vjLZF9yeL2yzEV_QsCzVRy8u69lRi4L9SQLLm7KN1H88ZSYGnYRv2)

The feature maps of the **first convolution layer** are shown below. The layer seems to detect some basic shapes such as lines and circles. For example, feature maps 0 and 3 have pixels activated in lines.

[https://lh4.googleusercontent.com/tZP8lL9OjwpVHm8ECt4nB4Dzvxh1hO4lr8TPiUYrahhwuF-oE2prTwaOwp3ts6yPtlQboaVt4YbubOPJ9qh1c6t2Q31Uexnn57zICS-qwrOTbqhDO6cM4_qtIs7kyUGaxJ2P_bXN](https://lh4.googleusercontent.com/tZP8lL9OjwpVHm8ECt4nB4Dzvxh1hO4lr8TPiUYrahhwuF-oE2prTwaOwp3ts6yPtlQboaVt4YbubOPJ9qh1c6t2Q31Uexnn57zICS-qwrOTbqhDO6cM4_qtIs7kyUGaxJ2P_bXN)

We plot the kernel that generates feature map 6 and 7. Kernel 6 seems to detect slashes in the image. Kernel 7 seems to detect curves in the image.

[https://lh6.googleusercontent.com/DxVn5nzPZAAe3uoydQJoXi95dnNMxRNIh5q_MI_x7N8i9hzfNA9ILtyB7sQdRVGfSiX-Uxnhfs88AtZe8nJ1gQW2sm9tQ0PLPpqfUJ6qfIqJ3bo5_JD4HDkRlmgu_ZCJ-E2PDqsJ](https://lh6.googleusercontent.com/DxVn5nzPZAAe3uoydQJoXi95dnNMxRNIh5q_MI_x7N8i9hzfNA9ILtyB7sQdRVGfSiX-Uxnhfs88AtZe8nJ1gQW2sm9tQ0PLPpqfUJ6qfIqJ3bo5_JD4HDkRlmgu_ZCJ-E2PDqsJ)

Kernel 6

[https://lh6.googleusercontent.com/KUkN_i38PUrBKbkm0HDZD6q6J9XqOllh7Hfv7P-bUrJam-2dfqXdnxBaheF-U2JPEFC4SSz-yLnuWu0fdOjaFiL5CcAlYAOkelZDRjUVZa4ylqOyHl5bM38rqHzcVBjSePd-3RQL](https://lh6.googleusercontent.com/KUkN_i38PUrBKbkm0HDZD6q6J9XqOllh7Hfv7P-bUrJam-2dfqXdnxBaheF-U2JPEFC4SSz-yLnuWu0fdOjaFiL5CcAlYAOkelZDRjUVZa4ylqOyHl5bM38rqHzcVBjSePd-3RQL)

Kernel 7

The feature maps of the **second convolution layer**, where we have 16 kernels, are shown below. The layer seems to detect more abstractive features. For example, feature map 2 was detecting some sort of features of features in the image.

[https://lh4.googleusercontent.com/ovrdGyf_MppaCcMpFFU7O0adGhrEVI8Uws9ws3c6i08RHqlPVh6EyFYHbdcNAL8-uNqTzTJ3rSyUvS8ICDvBk5AzT5uT-rrdCpNyhTqZesiTER8hHcv7LiApMGHz1Xx5vG5HytNb](https://lh4.googleusercontent.com/ovrdGyf_MppaCcMpFFU7O0adGhrEVI8Uws9ws3c6i08RHqlPVh6EyFYHbdcNAL8-uNqTzTJ3rSyUvS8ICDvBk5AzT5uT-rrdCpNyhTqZesiTER8hHcv7LiApMGHz1Xx5vG5HytNb)

---

## Task 4: CIFAR-10 Classification

We preprocessed the dataset with zero-center normalization  ($x' = \frac{x-\mu}{255.0}$) while loading the dataset with the function data_loader_cifar(). The mean is calculated based on the training data solely and then applied to the testing data. We store the mean of the training dataset for the visualization of the images.

The CIFAR-10 dataset contains relatively sophisticated images compared to the MNIST dataset. Hence, a deeper network with a larger capacity is needed. Also, we found that larger kernel/filter sizes perform better in this dataset since using smaller ones took a long time to converge. Max-pooling was used in this network for reducing the number of parameters and speeding up the training.

We built a convolutional neural network with three convolutional layers, each follows a max-pooling, and two fully connected layers. It took us more than eight hours to reach fair prediction results since we did not use parallel programming techniques in the package. The code of using building CNN with the self-implemented package is as follow:

```python
## build CNN
dnn = NN(lr=LEARNING_RATE, momentum=MOMENTUM, callback=history, clipping=False, weight_decay=False)
dnn.set_input_data(X_train)
dnn.set_target(y_train)

## Conv 1
dnn.add_conv2d_layer(16, kernel_size=(5,5), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=3)

## Conv 2
dnn.add_conv2d_layer(32, kernel_size=(5,5), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=3)

## Conv 3
dnn.add_conv2d_layer(64, kernel_size=(5,5), stride=1, activation='relu', padding='same')
dnn.add_maxpool2d_layer(kernel_size=3)

dnn.add_flatten_layer()
dnn.add_FC_layer(n_hidden=32, activation='relu')
dnn.add_FC_layer(n_hidden=10, activation=False)
dnn.add_stable_softmax_cross_entropy_layer()
```

[https://lh3.googleusercontent.com/UNOBuMtF-Ap0twqJNyqR7SotyUHia0pT8p0rGqs3Po7J2RakRKzmB0id26thTbme6zbpEqMhUAUBrTjRrAYsczp6RRGJJ7LUoGHjp8mlxU1ONOu9vTcMu8BQdFlxYgcE8utvhT3H](https://lh3.googleusercontent.com/UNOBuMtF-Ap0twqJNyqR7SotyUHia0pT8p0rGqs3Po7J2RakRKzmB0id26thTbme6zbpEqMhUAUBrTjRrAYsczp6RRGJJ7LUoGHjp8mlxU1ONOu9vTcMu8BQdFlxYgcE8utvhT3H)

[https://lh3.googleusercontent.com/6hPUFOSpKeHMF_I5FFN7NWoUAHyKkcCuKJW2SxjBnF6kR8NA89NmOPMmZaD8blyHQZYrltsiiCnYpMRdoMkFzdQun_4Kpv1AQAt5xxNjQTa3fbL533fygjRe5XnyWokbKHf3p1UI](https://lh3.googleusercontent.com/6hPUFOSpKeHMF_I5FFN7NWoUAHyKkcCuKJW2SxjBnF6kR8NA89NmOPMmZaD8blyHQZYrltsiiCnYpMRdoMkFzdQun_4Kpv1AQAt5xxNjQTa3fbL533fygjRe5XnyWokbKHf3p1UI)

### Prediction results

On the left-hand side, a cat was correctly predicted as a cat. However, on the right-hand side, a dog was wrongly predicted as a cat. The reason might be that the dog has brown ears, which is similar to the patterns on the cat. Since we have used zero-center to normalize the data, we added the offset of 0.47 (the average value of training data) back to the image to obtain the original image.

[https://lh3.googleusercontent.com/kO3eM16TAsA0JmWOU3R6OtjDZ7_ixncTqYMf3UqJYPE5ZG_-3yg76d3rW6sP3h45Yh1I1nlzOTHx9QEk30niv5vRnAUJTcoAlGxpDnCpSKsQShKFCgwHyNxduZvwM8EEIh-Oaqra](https://lh3.googleusercontent.com/kO3eM16TAsA0JmWOU3R6OtjDZ7_ixncTqYMf3UqJYPE5ZG_-3yg76d3rW6sP3h45Yh1I1nlzOTHx9QEk30niv5vRnAUJTcoAlGxpDnCpSKsQShKFCgwHyNxduZvwM8EEIh-Oaqra)

### Feature map visualization

We set the input as an image of a dog. The dog has two colors (brown and white) and the background is dark green.

[https://lh3.googleusercontent.com/XxzDJVlNuZNG_iW1dgaJtx5eiAB4mKakL6ga6N1JT77z--fPTAhoMVdi1S9XNAMag905z_iGWOiIQ8uO6u8AsPSV_q-jkxTiiuqyjMSOAMMPrXcmU4cHj0lkpKTVm17V0GNDbTO5](https://lh3.googleusercontent.com/XxzDJVlNuZNG_iW1dgaJtx5eiAB4mKakL6ga6N1JT77z--fPTAhoMVdi1S9XNAMag905z_iGWOiIQ8uO6u8AsPSV_q-jkxTiiuqyjMSOAMMPrXcmU4cHj0lkpKTVm17V0GNDbTO5)

The figure below shows the feature maps of the first convolutional layer. We can observe that some feature maps are detecting color patches. For example, kernel 0 was detecting brown patches and kernels 5 and 11 were detecting white patches.

[https://lh6.googleusercontent.com/4dUa-WhNMA71PZlBZbrdvY_FXm5929NVI0EhcBKqB0arIb_jlWJJ5x_UE918A9ZvQtYrkyKyCRug2O-ygRx3wtAAYz-yCaCMQc-6Kq2ADvwVO-4SkIYrWmd9SpUzGtTHP8iUgsxI](https://lh6.googleusercontent.com/4dUa-WhNMA71PZlBZbrdvY_FXm5929NVI0EhcBKqB0arIb_jlWJJ5x_UE918A9ZvQtYrkyKyCRug2O-ygRx3wtAAYz-yCaCMQc-6Kq2ADvwVO-4SkIYrWmd9SpUzGtTHP8iUgsxI)

The figure below shows the feature maps in the second layer. The feature maps seem to be more abstract than the first layer.

[https://lh5.googleusercontent.com/o-0D1oxhJm2Xio2EjwWWNQgShtAqh-0h5FQpOltERkOzqXneUpgXuRg2L1dclj3GxakvK6fy1iXw4H6HOk2M1xBqqSL6xLRq5nM8kcgkycFAXwM8ngCXHgU1wyVslcwVxUvSWGek](https://lh5.googleusercontent.com/o-0D1oxhJm2Xio2EjwWWNQgShtAqh-0h5FQpOltERkOzqXneUpgXuRg2L1dclj3GxakvK6fy1iXw4H6HOk2M1xBqqSL6xLRq5nM8kcgkycFAXwM8ngCXHgU1wyVslcwVxUvSWGek)