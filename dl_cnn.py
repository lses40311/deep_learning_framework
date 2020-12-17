import numpy as np
from numpy import linalg as LA
import time
import pickle
np.seterr(over='warn')
np.set_printoptions(precision=2)


class Variable:
    def __init__(self, name:str, type, value=None, operation=None):
        self.name = name
        self.type = type
        self.operation = operation ## op is a class
        if not value is None:
            self.value = np.array(value).astype(float)
        else:
            self.value = None
        if type == 'parameter':
            self.shape = self.value.shape
        else:
            self.shape = None


class NN:
    def __init__(self, lr=0.01, momentum=0.0, callback=None, clipping=False, weight_decay=False):
        self.lr = lr
        self.last_output = None
        self.loss_variable_name = 'z'
        self.last_output_dimension = None
        self.n_data = None
        self.n_layer = 0
        self.grad_table = {}
        self.G = ComputationalGraph()
        self.T = []
        self.velocity = {}
        self.momentum = momentum
        self.fc_weights = []
        self.weight_decay = weight_decay
        self.cnt_mid_node = 0
        self.callback = callback
        self.time_elapsed_batch = []
        self.clipping = clipping


    def predict(self, X):
        '''
        set input
        compute(output vertex) forwardpass
        '''
        self.G.put_input_data(X, None)
        self.G.clear_ops()
        self.G.compute(self.last_output)
        return self.G.vertex_container[self.last_output].value

    def accuracy(self, y_pred, y):
        n_total = float(y_pred.shape[0])
        true_labels = np.argmax(y, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)
        n_correct = np.sum(true_labels == pred_labels)
        return float(n_correct) / n_total

    def score(self, X, y):
        y_pred = self.predict(X)
        # print('pred {}, actual {}'.format(y_pred, y))
        return self.accuracy(y_pred, y)

    def return_scores(self, epoch, batch_ix, batch_score=0):
        print('Evaluating...')
        X_validate_lst, y_validate_lst = self.callback.get_testing_data()
        scores = [batch_score]
        for X_validate_batch, y_validate_batch in zip(X_validate_lst, y_validate_lst):
            scores.append(self.score(X_validate_batch, y_validate_batch))
        self.callback.record_score(epoch, batch_ix, scores, self.current_loss)
        print('Evaluation done. Accuracy {}, Loss {}'.format(scores, self.current_loss))

    def train(self, n_epoch: int, mini_batch_size: int):
        '''
        n_epoch: int
        mini_batch_size: int
        '''
        n_batch = int(self.n_data / mini_batch_size)

        ## initialize velocity
        for tunable_vertex_name in self.T:
            self.velocity[tunable_vertex_name] = np.zeros(self.G.vertex_container[tunable_vertex_name].shape)
        for epoch in range(n_epoch):
            for batch_ix, (X_batch, y_batch) in enumerate(self.iterate_minibatches(self.X, self.y, mini_batch_size, True)):
                start = time.time()
                print('-'*50)
                print('Epoch {}, batch {}/{}.'.format(epoch, batch_ix, n_batch))
                self.G.put_input_data(X_batch, y_batch)
                self.back_propagation()
                time_elapsed = time.time() - start
                print('Time elapsed: {:2f}.'.format(time_elapsed))
                self.time_elapsed_batch.append(time_elapsed)
                if not self.callback is None and batch_ix%5 == 0:
                    self.return_scores(epoch, batch_ix, self.score(X_batch, y_batch))
                    self.save()
            self.lr /= 2

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def back_propagation(self):
        ## get gradients and perform gradient decent
        self.grad_table = {self.loss_variable_name: np.array(1.0)}

        ## zero the parameter gradients
        self.G.clear_ops()
        start = time.time()

        ## forward pass
        self.G.compute(self.loss_variable_name)
        forward = time.time()
        self.current_loss = self.G.vertex_container[self.loss_variable_name].value
        print('Loss: {}'.format(self.current_loss))

        ## backpropogation
        for vertex_name in reversed(self.T):
            print('SGD for vertex {}'.format(vertex_name))
            self.build_grad(vertex_name)
            self.velocity[vertex_name] = self.momentum * self.velocity[vertex_name] - self.lr * self.grad_table[vertex_name]
            print("Gradient magnitude: {}".format(LA.norm(self.grad_table[vertex_name])))
            print("Velocity: {}".format(LA.norm(self.velocity[vertex_name])))
            # if vertex_name in self.fc_weights:
            #     print(self.velocity[vertex_name][0])
            #     print(self.G.vertex_container[vertex_name].value[0])
            if self.weight_decay and vertex_name in self.fc_weights:
                self.G.vertex_container[vertex_name].value -= self.weight_decay * self.G.vertex_container[vertex_name].value
            self.G.vertex_container[vertex_name].value += self.velocity[vertex_name]

        backward = time.time()
        print('Forward time {:2f}, backward time {:2f}.'.format(forward - start, backward - forward))


    def build_grad(self, vertex_name):
        if vertex_name in self.grad_table.keys():
            return self.grad_table[vertex_name]
        grads_of_path = []
        consumers = self.G.get_consumers(vertex_name)
        for variable in consumers:
            op = variable.operation
            D = self.build_grad(variable.name)
            start = time.time()
            grads_of_path.append(op.bprop(self.G.get_inputs(variable.name), self.G.vertex_container[vertex_name], D))
            # print('bprop for {} took {}.'.format(variable.name, time.time()-start))
        # print('save gradient for {} from {} paths'.format(vertex_name, len(grads_of_path)))
        # if vertex_name in self.fc_weights:
        #     print(grads_of_path)
        G = sum(grads_of_path)
        ## gradient clipping
        if(self.clipping and LA.norm(G) > self.clipping):
            print('Gradient norm {}: clipped at {}.'.format(LA.norm(G), self.clipping))
            G = self.clipping * (G / LA.norm(G))

        self.grad_table[vertex_name] = G
        return G

    def set_input_data(self, X: np.array):
        self.last_output = 'x'
        self.n_data = X.shape[0]
        self.last_output_dimension = X.shape
        self.X = X
        x = Variable('x', 'example', None)
        self.G.add_vertex(x)

    def set_target(self, y: np.array):
        self.target = 'y'
        self.y = y
        y = Variable('y', 'example', None)
        self.G.add_vertex(y)


    def get_temp_node_name(self):
        ret = self.cnt_mid_node
        self.cnt_mid_node += 1
        return 'm_' + str(ret)

    def add_FC_layer(self, n_hidden:int, activation='relu'):
        suffix = str(self.n_layer)
        W_init = np.random.randn(self.last_output_dimension[1], n_hidden) * np.sqrt(1.0/float(n_hidden)) ## FxH
        W = Variable('W_' + suffix, 'parameter', W_init)
        u1 = Variable('u1_' + suffix, 'op')

        b_init = np.zeros(n_hidden).reshape(1,-1) ## 1xH
        b = Variable('b_' + suffix, 'parameter', b_init)
        z = Variable('z_' + suffix, 'op')

        a = Variable('a_' + suffix, 'op')
        self.G.add_list_of_vertex([W, u1, b, z, a])

        self.G.add_edge(self.last_output, 'u1_' + suffix) # x -> u
        self.G.add_edge('W_' + suffix, 'u1_' + suffix) # W -> u
        self.G.set_operation('u1_' + suffix , 'matmul')

        self.G.add_edge('u1_' + suffix, 'z_' + suffix) # u -> z
        self.G.add_edge('b_' + suffix, 'z_' + suffix) # b -> z
        self.G.set_operation('z_' + suffix , 'plus')

        if activation == 'relu':
            self.G.add_edge('z_' + suffix, 'a_' + suffix) # z -> a
            self.G.set_operation('a_' + suffix , 'relu')
        elif not activation:
            self.G.add_edge('z_' + suffix, 'a_' + suffix) # z -> a
            self.G.set_operation('a_' + suffix , 'through')

        ## update the latest output shape
        self.last_output_dimension = (self.n_data, n_hidden)
        self.n_layer = self.n_layer + 1
        self.last_output = 'a_' + suffix
        self.T.append('W_' + suffix)
        self.T.append('b_' + suffix)
        self.fc_weights.append('W_' + suffix)

    def new_length(self, L, LL, S):
        return int((L - LL)/S + 1)

    def add_flatten_layer(self):
        suffix = str(self.n_layer)
        flat = Variable('f_' + suffix, 'op')
        self.G.add_list_of_vertex([flat])
        self.G.set_operation('f_' + suffix, 'flatten')
        self.G.add_edge(self.last_output, 'f_' + suffix) # a -> f

        n_feature = self.last_output_dimension[1]*self.last_output_dimension[2]*self.last_output_dimension[3]
        self.last_output_dimension =  (self.n_data, n_feature)## updated to F
        self.n_layer = self.n_layer + 1
        self.last_output = 'f_' + suffix

    def add_conv2d_layer(self, n_kernel, kernel_size=(3,3), stride=1, activation='relu', padding='same'):
        ## K_var.stride, K_var.padding implement
        suffix = str(self.n_layer)
        ## self.last_output_dimension == (N, F, H, W)
        n_input_neurons = kernel_size[0] * kernel_size[1] * self.last_output_dimension[1]
        K_init = np.random.randn(n_kernel, self.last_output_dimension[1], kernel_size[0], kernel_size[1]) * np.sqrt(1.0/float(n_input_neurons)) # * np.sqrt(1.0/float(n_kernel * kernel_size[0]* kernel_size[1]))
        K = Variable('K_' + suffix, 'parameter', K_init)
        K.stride = stride
        K.padding = padding
        u1 = Variable('u1_' + suffix, 'op')

        b_init = np.zeros((1, n_kernel, 1, 1)) # (1,C,1,1)
        b = Variable('b_' + suffix, 'parameter', b_init)
        z = Variable('z_' + suffix, 'op')

        a = Variable('a_' + suffix, 'op')
        self.G.add_list_of_vertex([K, u1, b, z, a])

        self.G.add_edge(self.last_output, 'u1_' + suffix) # x -> u
        self.G.add_edge('K_' + suffix, 'u1_' + suffix) # K -> u
        self.G.set_operation('u1_' + suffix , 'conv2d')

        self.G.add_edge('u1_' + suffix, 'z_' + suffix) # u -> z
        self.G.add_edge('b_' + suffix, 'z_' + suffix) # b -> z
        self.G.set_operation('z_' + suffix , 'plus')

        if activation == 'relu':
            self.G.add_edge('z_' + suffix, 'a_' + suffix) # z -> a
            self.G.set_operation('a_' + suffix , 'relu')
        elif not activation:
            self.G.add_edge('z_' + suffix, 'a_' + suffix) # z -> a
            self.G.set_operation('a_' + suffix , 'through')

        ## update the latest output shape
        if padding == 'same':
            (H_new, W_new) = (self.last_output_dimension[2], self.last_output_dimension[3])
        elif padding == 'valid':
            (H_new, W_new) = (self.new_length(self.last_output_dimension[2], kernel_size[0], stride),
            self.new_length(self.last_output_dimension[3], kernel_size[1], stride))
        elif padding == 'full':
            (H_new, W_new) = (self.new_length(self.last_output_dimension[2]+2*kernel_size[0]-1, kernel_size[0], stride),
            self.new_length(self.last_output_dimension[3]+2*kernel_size[1]-1, kernel_size[1], stride))
        else:
            raise('No such padding.')
        self.last_output_dimension = (self.n_data, n_kernel, H_new, W_new) ## updated to (F, H', W')
        self.n_layer = self.n_layer + 1
        self.last_output = 'a_' + suffix
        self.T.append('K_' + suffix)
        self.T.append('b_' + suffix)
        self.fc_weights.append('K_' + suffix)


    def add_maxpool2d_layer(self, kernel_size=2):
        stride = kernel_size
        p = Variable('pool' + str(self.n_layer), 'op')
        self.G.add_vertex(p)
        self.G.add_edge(self.last_output, 'pool' + str(self.n_layer))
        self.G.set_operation('pool' + str(self.n_layer), 'maxpool2d', size=kernel_size, stride=stride)
        (H_new, W_new) = (self.new_length(self.last_output_dimension[2], kernel_size, stride),
            self.new_length(self.last_output_dimension[3], kernel_size, stride))
        self.last_output_dimension = (self.n_data, self.last_output_dimension[1], H_new, W_new) ## updated to (F, H', W')
        self.last_output = 'pool' + str(self.n_layer)
        self.n_layer = self.n_layer + 1

    def add_mse_layer(self):
        mse = Variable('mse', 'op')
        self.G.add_vertex(mse)
        self.G.set_operation('mse', 'mse')
        self.G.add_edge(self.last_output, 'mse')
        self.G.add_edge(self.target, 'mse')

        z = Variable('z', 'op')
        self.G.add_vertex(z)
        self.G.add_edge('mse', 'z')
        self.G.set_operation('z', 'summation')

    def add_softmax_layer(self):
        s = Variable('s', 'op')
        self.G.add_vertex(s)
        self.G.add_edge(self.last_output, 's')
        self.G.set_operation('s', 'softmax')
        self.last_output = 's'

    def add_cross_entropy_layer(self):
        z = Variable('z', 'op')
        self.G.add_vertex(z)
        self.G.add_edge(self.last_output, 'z')
        self.G.add_edge(self.target, 'z')
        self.G.set_operation('z', 'cross_entropy')
        self.add_softmax_layer()

    def add_stable_softmax_cross_entropy_layer(self):
        z = Variable('z', 'op')
        self.G.add_vertex(z)
        self.G.add_edge(self.last_output, 'z')
        self.G.add_edge(self.target, 'z')
        self.G.set_operation('z', 'softmax_crossentropy')

        ## for prediction
        ## identical code as func add_softmax_layer, not a consumer
        s = Variable('s', 'op')
        self.G.add_vertex(s)
        self.G.add_edge(self.last_output, 's', is_consumer=False)
        self.G.set_operation('s', 'softmax')
        self.last_output = 's'

    def add_weight_decay(self, penalty_weight=0.001):
        lamb = Variable('lambda', 'example')
        lamb.value = penalty_weight
        self.G.add_vertex(lamb)

        l = Variable('l', 'op')
        self.G.add_vertex(l)
        self.G.set_operation('l', 'plus')
        self.G.add_edge(self.loss_variable_name, 'l')
        self.loss_variable_name = 'l'
        for vertex_name in self.fc_weights:
            ## calculate l2 norm of the weight
            l2_name = self.get_temp_node_name()
            l2 = Variable(l2_name, 'op')
            self.G.add_vertex(l2)
            self.G.add_edge(vertex_name, l2_name)
            self.G.set_operation(l2_name, 'l2')

            ## multiply by the lambda
            weighted_l2_name = self.get_temp_node_name()
            weighted_l2 = Variable(weighted_l2_name, 'op')
            self.G.add_vertex(weighted_l2)
            self.G.add_edge(l2_name, weighted_l2_name)
            self.G.add_edge('lambda', weighted_l2_name)
            self.G.set_operation(weighted_l2_name, 'dot')

            ## add the penalty to the loss function
            self.G.add_edge(weighted_l2_name, 'l')

    def save(self):
        with open("dl_class.pickle", 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

class ComputationalGraph:
    def __init__(self, vertex_lst=[]):
        self.vertex_container = {}
        self.edges_consumers = {}
        self.edges_inputs = {}
        self.add_list_of_vertex(vertex_lst)

    def add_vertex(self, v:Variable):
        if v.name in self.vertex_container:
            print("Vertex name already exists: {}".format(v.name))
        else:
            self.vertex_container[v.name] = v
            self.edges_consumers[v.name] = []
            self.edges_inputs[v.name] = []

    def add_list_of_vertex(self, vertex_lst):
        for vertex in vertex_lst:
            self.add_vertex(vertex)

    def put_input_data(self, X, y):
        self.vertex_container['x'].value = X
        self.vertex_container['y'].value = y

    def print_vertex(self):
        for key, variable in self.vertex_container.items():
            print("{:2} :type={:9}, shape={}, op={}.".format(key, variable.type, variable.value.shape, variable.operation))

    def add_edge(self, from_name:str, to_name:str, is_consumer=True):
        ## check if exists
        assert (from_name in self.vertex_container) and (to_name in self.vertex_container)
        if to_name in self.edges_consumers[from_name]:
            print("Edge exists. ({} -> {})".format(from_name, to_name))
        else:
            if is_consumer:
                self.edges_consumers[from_name].append(to_name)
            self.edges_inputs[to_name].append(from_name)

    def print_edges(self):
        for key, input_lst in self.edges_inputs.items():
            print("{:15} -> {:4}: {}".format(str(input_lst), key, self.vertex_container[key].operation))
            if not self.vertex_container[key].value is None:
                try:
                    print(" "* 19 + str(self.vertex_container[key].value.shape))
                except:
                    pass


    def get_operation(self, vertex_name:str):
        return vertex_container[vertex_name].operation


    def get_consumers(self, vertex_name:str):
        '''
        Return a list of variables that consume the vertex.
        '''
        return [self.vertex_container[v_name] for v_name in self.edges_consumers[vertex_name]]


    def get_inputs(self, vertex_name:str):
        return [self.vertex_container[v_name] for v_name in self.edges_inputs[vertex_name]]


    def clear_ops(self):
        for key, variable in self.vertex_container.items():
            if variable.type == 'op':
                variable.value = None

    def compute(self, vertex_name):
        if self.vertex_container[vertex_name].value is None:
            input_variable_lst = []
            for input in self.edges_inputs[vertex_name]:
                self.compute(input)
                input_variable_lst.append(self.vertex_container[input])
            # print("computing value for vertex {}".format(vertex_name))
            # print('input 0 {}'.format(input_variable_lst[0].value[0]))
            # self.vertex_container[vertex_name].value = self.vertex_container[vertex_name].operation.f([self.vertex_container[v_name] for v_name in self.edges_inputs[vertex_name]])
            self.vertex_container[vertex_name].value = self.vertex_container[vertex_name].operation.f(input_variable_lst)

    def set_operation(self, vertex_name, operation, **kwargs):
        if operation == 'plus':
            self.vertex_container[vertex_name].operation = op_plus()
        elif operation == 'matmul':
            self.vertex_container[vertex_name].operation = op_matmul()
        elif operation == 'relu':
            self.vertex_container[vertex_name].operation = op_relu()
        elif operation == 'mse':
            self.vertex_container[vertex_name].operation = op_mse()
        elif operation == 'dot':
            self.vertex_container[vertex_name].operation = op_dot()
        elif operation == 'summation':
            self.vertex_container[vertex_name].operation = op_summation()
        elif operation == 'cross_entropy':
            self.vertex_container[vertex_name].operation = op_cross_entropy()
        elif operation == 'softmax':
            self.vertex_container[vertex_name].operation = op_softmax()
        elif operation == 'through':
            self.vertex_container[vertex_name].operation = op_through()
        elif operation == 'l2':
            self.vertex_container[vertex_name].operation = op_l2()
        elif operation == 'flatten':
            self.vertex_container[vertex_name].operation = op_flatten()
        elif operation == 'conv2d':
            self.vertex_container[vertex_name].operation = op_conv2d()
        elif operation == 'softmax_crossentropy':
            self.vertex_container[vertex_name].operation = op_stable_softmax_cross_entropy()
        elif operation == 'maxpool2d':
            self.vertex_container[vertex_name].operation = op_max_pool_2d(kwargs['size'], kwargs['stride'])
        else:
            print("Operation {} not defined.".format(operation))

class op_plus:
    def f(self, input_lst): ## list of Variables
        # a = input_lst[0].value
        # b = input_lst[1].value
        return sum([variable.value for variable in input_lst])

    def bprop(self, input_lst, wrt_variable, G):
        # a = input_lst[0]
        # b = input_lst[1]
        if wrt_variable in input_lst:
            reduce_axes = ()
            for axis, (var_axis_size, G_axis_size) in enumerate(zip(wrt_variable.value.shape, G.shape)):
                # print('var {}, G {}'.format(var_axis_size, G_axis_size))
                if var_axis_size != G_axis_size:
                    reduce_axes += (axis,)
            if len(reduce_axes) > 0:
                G = np.sum(G, axis=reduce_axes, keepdims=True)
            return G
            ## Assignment 1's working solution
            # while np.ndim(G) > len(wrt_variable.value.shape):
            #     G = np.sum(G, axis=0)
            # return G * np.ones(wrt_variable.value.shape)

        # if wrt_variable.name == a.name:
        #     while np.ndim(G) > len(a.value.shape):
        #         G = np.sum(G, axis=0)
        #     return G * np.ones(a.value.shape)
        # elif wrt_variable.name == b.name:
        #     while np.ndim(G) > len(b.value.shape):
        #         G = np.sum(G, axis=0)
        #     return G * np.ones(b.value.shape)
        else:
            print("wrt not in the inputs.")
            assert 1 == 0

## for mat*mat and mat*vector
class op_matmul:
    def f(self, input_lst):
        A = input_lst[0].value
        B = input_lst[1].value
        return np.matmul(A, B)

    def bprop(self, input_lst, wrt_variable, G):
        A_var = input_lst[0]
        B_var = input_lst[1]
        A = A_var.value
        B = B_var.value
        # print('G {}, B.T {}'.format(G.shape, B.T.shape))
        if wrt_variable.name == A_var.name:
            return np.matmul(G, B.T)
        elif wrt_variable.name == B_var.name:
            # print("AT {}, G {}".format(A.T, G))
            return np.matmul(A.T, G)
        else:
            print("wrt not in the inputs.")
            return 0

class op_dot:
    def f(self, input_lst):
        a = input_lst[0].value
        b = input_lst[1].value
        return np.dot(a, b)

    def bprop(self, input_lst, wrt_variable, G):
        A_var = input_lst[0]
        B_var = input_lst[1]
        if wrt_variable.name == A_var.name:
            return G * B_var.value
        elif wrt_variable.name == B_var.name:
            return G * A_var.value
        else:
            print("wrt not in the inputs.")
            return 0

class op_relu:
    def f(self, input_lst):
        A = input_lst[0].value
        return A * (A > 0)

    def bprop(self, input_lst, wrt_variable, G):
        A = input_lst[0].value
        return (1. * (A > 0)) * G

class op_mse:
    def f(self, input_lst):
        A = input_lst[0].value
        B = input_lst[1].value
        return abs(A-B) ** 2

    def bprop(self, input_lst, wrt_variable, G):
        A = input_lst[0].value
        B = input_lst[1].value
        return G * (A-B)

class op_summation:
    def f(self, input_lst):
        a = input_lst[0].value
        return np.sum(a)

    def bprop(self, input_lst, wrt_variable, G):
        a = input_lst[0].value
        return G * np.ones((a.shape))

class op_through:
    def f(self, input_lst):
        return input_lst[0].value

    def bprop(self, input_lst, wrt_variable, G):
        a = input_lst[0].value
        return G * np.ones((a.shape))

class op_cross_entropy:
    def f(self, input_lst):
        A = input_lst[0].value
        B = input_lst[1].value
        return -np.sum(B * np.log(A))
        # print(A[B>0])
        # return -np.sum(np.log(A[B>0]))

    def bprop(self, input_lst, wrt_variable, G):
        a_var = input_lst[0]
        b_var = input_lst[1]
        if wrt_variable.name == a_var.name:
            return -G * (b_var.value / a_var.value)
        elif wrt_variable.name == b_var.name:
            return -G * (a_var.value / b_var.value)
        else:
            print("wrt not in the inputs.")
            return None

class op_softmax:
    def f(self, input_lst):
        A = input_lst[0].value
        return np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)
        # m = np.max(A, axis=1, keepdims=True)
        # return np.exp(A-m) / np.sum(np.exp(A-m), axis=1, keepdims=True)

    def row_of_jacobian(self, v, G_i):
        return np.matmul(G_i, ((np.identity(len(v)) - v.reshape((-1,1))) * v.reshape((1,-1))))

    def bprop(self, input_lst, wrt_variable, G):
        A = self.f(input_lst)
        jacobian = np.array([self.row_of_jacobian(row, G[i]) for i, row in enumerate(A)])
        return jacobian
        # jacobian = (np.identity(len(v)) - v.reshape((-1,1))) * v.reshape((1,-1))
        # return np.matmul(G.reshape((1,-1)), jacobian)

class op_stable_softmax_cross_entropy:
    def f(self, input_lst):
        A_var = input_lst[0]
        T_var = input_lst[1]
        A = A_var.value
        T = T_var.value
        m = np.max(A, axis=1, keepdims=True)
        return -np.sum(T * (A - m - np.log(np.sum(np.exp(A-m), axis=1, keepdims=True))))


    def bprop(self, input_lst, wrt_variable, G):
        A_var = input_lst[0]
        T_var = input_lst[1]
        A = A_var.value
        T = T_var.value
        m = np.max(A, axis=1, keepdims=True)
        Y_hat = np.exp(A-m) / np.sum(np.exp(A-m), axis=1, keepdims=True)
        return G * (Y_hat - T)

class op_l2:
    def f(self, input_lst):
        return LA.norm(input_lst[0].value)

    def bprop(self, input_lst, wrt_variable, G):
        grad = G * input_lst[0].value
        return grad

class op_flatten:
    def f(self, input_lst):
        tensor_var = input_lst[0]
        return tensor_var.value.reshape(tensor_var.value.shape[0], -1)
        # self.value_shape = tensor_var.value.shape
        # return tensor_var.value.reshape(self.value_shape[0], -1)

    def bprop(self, input_lst, wrt_variable, G):
        return G.reshape(input_lst[0].value.shape)

class op_conv2d:
    def get_pad_wh(self, pads):
        ## return (pad_top, pad_bottom), (pad_left, pad_right)
        pad_H = (int(np.ceil(pads[0])),int(np.floor(pads[0])))
        pad_W = (int(np.ceil(pads[1])),int(np.floor(pads[1])))
        return pad_H, pad_W

    def new_length(self, L, LL, S):
        return int((L - LL)/S + 1)

    def zero_padding(self, X, padding):
        ## pads[0] pad Height, pads[1] pad Width
        if padding == 'same':
            pad_H, pad_W = self.get_pad_wh(((self.HH-1)/2, (self.WW-1)/2))
        elif padding == 'full':
            pad_H, pad_W = self.get_pad_wh((self.HH-1, self.WW-1))
        elif padding == 'valid':
            pad_H = (0, 0)
            pad_W = (0, 0)
        else:
            pad_H = (padding, padding)
            pad_W = (padding, padding)
        self.pad_H = pad_H
        self.pad_W = pad_W
        return np.pad(X, ((0,0),(0,0),pad_H,pad_W), 'constant')

    def im2row_batch(self, X, C, HH, WW): ## for n images
        (N, C, H, W) = X.shape
        H_new = self.new_length(H, HH, self.stride)
        W_new = self.new_length(W, WW, self.stride)

        ## locate every pin in the image tensor
        pin_indeces = [(i, j) for i in np.arange(0, (H-HH)+1, self.stride) for j in np.arange(0, (W-WW)+1, self.stride)]

        n_pixel = C * HH * WW
        n_location = H_new * W_new
        assert len(pin_indeces) == n_location
        # print('n_location per image {}, n_pixel per cube of interest {}'.format(n_location, n_pixel))

        X_streched = np.zeros((N*n_location, n_pixel))
        for image_ix in range(N):
            for location_ix, pin_point in enumerate(pin_indeces):
                X_streched[image_ix*n_location+location_ix, :] = X[image_ix, :, pin_point[0]:(pin_point[0]+HH), pin_point[1]:(pin_point[1]+WW)].reshape(1,-1)

        return X_streched

    def convolution_batch(self, X, K, S, padding='valid'):
        '''
        X: 4 dimensional tensor (N, C, H, W)
        return: (N, F, H', W')
        '''
        # print('X shape: {}'.format(X.shape))
        # print('Kernel shape: {}'.format(K.shape))
        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)
        pad_X = self.zero_padding(X, padding)
        (F, CW, HH, WW) = K.shape
        (N, C, H, W) = pad_X.shape
        self.pad_x_shape = pad_X.shape
        H_new = self.new_length(H, HH, S)
        W_new = self.new_length(W, WW, S)
        assert C == CW
        X_streched = self.im2row_batch(pad_X, CW, HH, WW)


        # print('X strech shape {}, Kernel flatten shape {}'.format(X_streched.shape, K.transpose((1,2,3,0)).reshape(-1, F).shape))
        feature_map = X_streched @ K.transpose((1,2,3,0)).reshape(-1, F)
        # feature_map = X_streched @ K.reshape(F, -1).T

        # print(X[0])
        # print('-'*20)
        # print(feature_map[0])
        # print('-'*50)

        return feature_map.reshape(N, H_new, W_new, F).transpose(0,3,1,2)

    def f(self, input_lst):
        X_var = input_lst[0]
        K_var = input_lst[1]
        self.stride = K_var.stride
        self.padding = K_var.padding
        (self.N, self.C, self.H, self.W) = X_var.value.shape
        (self.F, self.CF, self.HH, self.WW) = K_var.value.shape
        ret = self.convolution_batch(X_var.value, K_var.value, K_var.stride, K_var.padding)
        # print(K_var.value[0])
        return ret


    def bprop(self, input_lst, wrt_variable, G):
        X_var = input_lst[0]
        K_var = input_lst[1]
        (N, F, H_new, W_new) = G.shape
        if wrt_variable == X_var:
            ## dO/dX
            # grad = np.zeros(X_var.value.shape)
            grad = np.zeros(self.pad_x_shape)

            for n in range(self.N):
                for f in range(self.F):
                    for h_out in range(H_new):
                        for w_out in range(W_new):
                            grad[n,:,(h_out*self.stride):(h_out*self.stride+self.HH),(w_out*self.stride):(w_out*self.stride+self.WW)] += G[n,f,h_out,w_out] * K_var.value[f,:,:,:]


            return grad[:,:,self.pad_H[0]:(self.pad_H[0]+self.H),self.pad_W[0]:(self.pad_W[0]+self.W)]

            # for n in range(self.N):
            #     for c in range(self.C):
            #         ## full padding?
            #         grad[n,c,:,:] += self.convolution_batch(G[np.newaxis,n,:,:,:], K_var.value[np.newaxis,:,c,:,:], S=1, padding='same').reshape((H_new, W_new))
            # return grad

            # not working
            # grad = np.zeros((self.N, self.C, self.H, self.W))
            # for n in range(N):
            #     grad[n,:,:,:] = self.convolution_batch(G[n,:,:,:], np.flip(K_var.value, axis=(2,3)).transpose((1,0,2,3)), S=1, padding=int((self.HH-1)/2))
            # return grad


            # for n in range(N):
            #     grad[n,:,:,:] = self.convolution_batch(G[n,:,:,:], np.flip(K_var.value, axis=(2,3)).transpose((1,0,2,3)), S=1, padding='same')
            # return grad
        elif wrt_variable == K_var:
            ## dO/dK
            ## return (F, C, HH, WW) gradient for K
            grad = np.zeros(K_var.value.shape)
            # G_reshaped = G.transpose(1, 2, 3, 0).reshape(F, -1)
            # grad = (G_reshaped @ self.X_streched).reshape((F, C, HH, WW))

            ## working
            pad_X = self.zero_padding(X_var.value, self.padding)
            for n in range(self.N):
                for f in range(self.F):
                    for h_out in range(H_new):
                        for w_out in range(W_new):
                            # print('Calculating grad form {},{},{},{}.'.format(n,f,h_out,w_out))
                            # print(pad_X[n,:,(h_out*self.stride):(h_out*self.stride+self.HH),(w_out*self.stride):(w_out*self.stride+self.WW)])
                            grad[f,:,:,:] += G[n,f,h_out,w_out] * pad_X[n,:,(h_out*self.stride):(h_out*self.stride+self.HH),(w_out*self.stride):(w_out*self.stride+self.WW)]
            return grad

            # # ## working?
            # for n in range(self.N):
            #     pad_X = self.zero_padding(X_var.value[np.newaxis,n,:,:,:], self.padding)
            #     for f in range (self.F):
            #         grad_from_one_feature_map = self.im2row_batch(pad_X, self.CF, self.HH, self.WW) * G[n,f,:,:].reshape((1,-1)).T
            #         grad[f,:,:,:] += np.sum(grad_from_one_feature_map, axis=0).reshape((self.CF, self.HH, self.WW))
            # return grad

            # for f in range (F):
            #     for c in range(C):
            #         ## sum is to add up the gradient from all examples
            #         # grad_example = self.convolution_batch(X_var.value[:,np.newaxis,c,:,:], np.expand_dims(G[:,f,:,:], axis=1), S=1, padding='record')
            #
            #         # grad_example = self.convolution_batch(X_var.value[:,np.newaxis,c,:,:], G[:,np.newaxis,f,:,:], S=1, padding='record')
            #         for X_slice, G_slice in zip(X_var.value[:,np.newaxis,np.newaxis,c,:,:], G[:,np.newaxis,np.newaxis,f,:,:]):
            #             grad_example = self.convolution_batch(X_slice, G_slice, S=1, padding='record').reshape((HH, WW))
            #             # print('got shape {}'.format(grad_example.shape))
            #             grad[f, c, :, :] += grad_example
            #         # grad[f, c, :, :] = np.sum(grad_example, axis=0, keepdims=True)
            # return grad


class op_max_pool_2d:
    def __init__(self, size, stride):
        self.size = size

        ## non overlapped
        self.stride = size

    def f(self, input_lst):
        X = input_lst[0].value
        (N, C, H, W) = X.shape
        H_new = int((H-self.size)/self.stride +1)
        W_new = int((W-self.size)/self.stride +1)
        down_sampled = np.zeros((N,C,H_new,W_new))
        # self.indices = []
        self.coordinates = np.zeros((N, C, H, W))
        pin_indices = [(i, j) for i in np.arange(0, H-self.size+1, self.stride) for j in np.arange(0, W-self.size+1, self.stride)]
        for n in range(N):
            for c in range(C):
                for i, j in pin_indices:
                    # print('inspecting coordinate ({},{},{},{})'.format(n,c,i,j))
                    mat = X[n,c,i:(i+self.size),j:(j+self.size)]
                    # print(mat)

                    (coor_h, coor_w) = np.unravel_index(np.argmax(mat), mat.shape)
                    (max_h, max_w) = (coor_h + i, coor_w + j)
                    # self.indices.append((max_h, max_w))
                    self.coordinates[n,c,max_h,max_w] = 1
                    # print('pooled {},{},{},{}'.format(n,c,max_h,max_w))
                    # print('saved index {},{},{},{}'.format(n,c,int(i/self.stride),int(j/self.stride)))
                    down_sampled[n,c,int(i/self.stride),int(j/self.stride)] = X[n,c,max_h,max_w]
        return down_sampled

    def bprop(self, input_lst, wrt_variable, G):
        X = input_lst[0].value
        (N, F, H_new, W_new) = G.shape
        (N, C, H, W) = X.shape
        grad = np.zeros((N, C, H, W))
        G_repeat = G.repeat(self.size, axis=2).repeat(self.size, axis=3)
        G_repeat =  np.pad(G_repeat, ((0,0),(0,0),(0,H-G_repeat.shape[2]),(0,W-G_repeat.shape[3])), 'constant')
        grad = self.coordinates * G_repeat
        # cnt = 0
        # for n in range(N):
        #     for f in range(F):
        #         for h_new in range(H_new):
        #             for w_new in range(W_new):
        #                 (h,w) = self.indices[cnt]
        #                 grad[n,f,h,w] += G[n,f,h_new,w_new]
        #                 cnt += 1
        return grad
