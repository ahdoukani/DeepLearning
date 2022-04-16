
"""
A neural network is composed of layers
There are many types of layers but each has can perform:
forward propagation - prediction based on inputs and existing parameters
inputs are passed forwards through layers in forwards propagation

backward progation- learning - calculating gradients for us in adjusting current parameters
 ... to minimize error in prediction

gradients are passed forwards though layers in back propagation

The following classes have been coded by :Joel Grus  https://github.com/joelgrus/joelnet.

1) layer base class,
2) linear layer,
3) activation base class
4) Tanh activation layer/ tanh prime

The following classes and functions are coded by myself :

1) Convolution layer class - inherits from Layer base class
2) Batch Normalization layer class - inherits from Layer base class
3) Sigmoid Activation layer class - inherits from activation class and layer base class
4) RelU, Leaky RelU Activation layer class  - inherits from activation class and layer base class /
 RelU and leaky RelU primes function
5) Flatten layer - inherits from layer base class / Flatten layer prime / flatten inputs function
6) pooling layer - inherits from layer base class - Only MAX pooling implemented but can be easily extended to include
.. average pooling function / Pooling layer prime
7) SoftMax activation- inherits from activation class and layer base class/ Softmax prime
8) padding function


"""
#  Dict and callable used to specify dict generic as a type hint

from typing import Dict, Callable

import numpy as np
from tensor import Tensor
from typing import Sequence



class Layer:
    """
    base class for layers
    """
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError


def flatten_inputs(inputs: Tensor) -> Tensor:
    n_images = np.shape(inputs)[0]
    input_rows = np.shape(inputs)[-2]
    input_cols = np.shape(inputs)[-1]
    shape_length = len(np.shape(inputs))

    match shape_length:
        case 4:
            channels = np.shape(inputs)[-3]
        case 3:
            channels = 1

        case shape_length if shape_length > 4:

            flat_channel = 1
            channels = np.shape(inputs)[1:-2]
            for i in range(0, len(channels)):
                flat_channel *= channels[i]

            channels = flat_channel

        case _: raise Exception("input size too small to flatten")

    f_inputs = inputs.reshape(n_images, channels * input_rows * input_cols)

    return f_inputs


class Linear(Layer):
    """
     A linear layer computes
    # output = input @ w + b
    # computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int, flatten_input=False) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()

        self.flatten_input = flatten_input
        self.input_size = input_size
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

        print('--------------------Linear layer of neural net created with random weight and bias-----------------')

    def forward(self, inputs: Tensor) -> Tensor:

        print('-----------------forward propagation for a linear layer just ran')
        print('--------------------outputs = inputs @ w + b --------------------')
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs

        if self.flatten_input:
          self.inputs = flatten_inputs(self.inputs)

        output = self.inputs @ self.params["w"] + self.params["b"]

        return output

    def backward(self, grad: Tensor) -> Tensor:

        print('-----------------backwards propagation for a linear layer just ran')

        self.grads["b"] = np.sum(grad, axis=0)

        self.grads["w"] = np.zeros(np.size(self.params["w"]))

        self.grads["w"] = self.inputs.T @ grad

        return grad @ self.params["w"].T

# F of type that can be called, it takes tensor and returns a tensor
# F can be used for type hinting
F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
        An activation layer just applies a function
        elementwise to its inputs
        """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        print('--------------------forward propagation for activation layer run-----------------')
        self.inputs = inputs

        # print('Inputs to linear layer: ', self.inputs)
        # print('outputs from linear layer: ', self.f(inputs))

        return self.f(self.inputs)

    def backward(self, grad: Tensor) -> Tensor:

        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        print('--------------------backward propagation for activation layer running-----------------')

        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:

    print('--------------------Tanh activation layer working-----------------')
    # print('tanh(output from linear layer):', np.tanh(x))
    output = np.tanh(x)

    return output



def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    print('--------------------calculating grad of Tanh()-----------------')

    print('--------------------Fished calculating grad of Tanh()----------')
    return 1 - y ** 2


class Tanh(Activation):
    def __init__(self):
        print('--------------------Activation layer constuctor activated---------')
        print('--------------------Tanh activation layer created-----------------')
        super().__init__(tanh, tanh_prime)


def relu(x: Tensor) -> Tensor:
    print('--------------------relu activation function started-----------------')

    print('--------------------relu activation function finished-----------------')
    return np.maximum(0, x)


def relu_prime(x: Tensor) -> Tensor:
    print('--------------------calculating grad of Relu(output from linear layer)-----------------')
    y = relu(x)
    y[y <= 0] = 0
    y[y > 0] = 1
    print('--------------------backward propagation for Relu activation layer just Finished--------')
    return y


class Relu(Activation):
    def __init__(self):
        print('--------------------Relu layer constuctor activated---------------')
        print('--------------------Relu activation layer created-----------------')
        super().__init__(relu, relu_prime)


def leaky_relu(x: Tensor, constant: float = 0.05) -> Tensor:
    print('--------------------Leaky relU activation layer started calculation-----------------')
    print('--------------------Leaky relU activation layer finished calculation-----------------')
    return np.maximum(constant*x, x)


def leaky_relu_prime(x: Tensor, constant: float = 0.05) -> Tensor:
    print('--------------------calculating grad of Leaky_Relu(output from linear layer)-----------------')
    x[x <= 0] = constant
    x[x > 0] = 1
    print('--------------------backward propagation for Leaky RelU activation layer just Finished-------')
    return x


class LeakyRelu(Activation):

    def __init__(self):
        print('--------------------Activation layer constuctor activated-----------------')
        print('--------------------Leaky Relu activation layer created-----------------')
        super().__init__(leaky_relu, leaky_relu_prime)


def sigmoid(x: Tensor) -> Tensor:

    return 1/(1+np.exp(-x))


def sigmoid_prime(x: Tensor) -> Tensor:
    y = sigmoid(x)

    return y*(1-y)


class Sigmoid(Activation):

    def __init__(self):
        print('--------------------Activation layer constuctor activated-----------------')
        print('--------------------Sigmoid activation layer created-----------------')
        super().__init__(sigmoid, sigmoid_prime)


def pad(inputs: Tensor, padding: tuple[int, int]):

    n_pad = [[0, 0] for dim in range(len(np.shape(inputs)))]
    n_pad[-2] = n_pad[-1] = [padding[0], padding[1]]
    n_pad_tuple = tuple(n_pad)

    padded_inputs = np.pad(inputs, n_pad_tuple, mode='constant', constant_values=(0, 0))

    return padded_inputs


def calculate_padding(input_size: tuple[int, ...],
                      filter_size: tuple[int, ...],
                      output_size: tuple[int, ...],
                      r_stride: int,
                      c_stride: int) -> tuple[int, int]:
    row_padding = int(((r_stride*(output_size[-2]-1))+filter_size[-2]-input_size[-2])/2)
    col_padding = int(((c_stride*(output_size[-1]-1))+filter_size[-1]-input_size[-1])/2)

    return row_padding, col_padding


class Convolution(Layer):

    def __init__(self, input_size: tuple[int, ...],
                 filters: Sequence[Tensor],
                 r_stride: int = 1,
                 c_stride: int = 1,
                 padding: int = 0,
                 ):

        super().__init__()
        self.input_size = input_size
        self.filters = filters
        self.f_rows = np.shape(self.filters[0])[-2]
        self.f_cols = np.shape(self.filters[0])[-1]
        self.n_filters = len(self.filters)
        self.r_stride = r_stride
        self.c_stride = c_stride
        self.padding = padding
        self.output_size = self.calc_output_size(input_size, (self.f_rows, self.f_cols), self.r_stride, self.c_stride,
                                                 self.padding)

        self.params["w"] = np.array(self.filters)
        self.params["b"] = np.random.randn(self.n_filters)
        check_dims_params(self.input_size, self.output_size, self.r_stride, self.c_stride)

    def calc_output_size(self, input_size: tuple[int, ...],
                         filter_size: tuple[int, ...],
                         r_stride: int = 1,
                         c_stride: int = 1,
                         padding: int = 0
                         ) -> tuple[int, ...]:

        o_rows: int = int(((input_size[-2] - filter_size[-2] + (2 * padding)) / r_stride) + 1)
        o_cols: int = int(((input_size[-1] - filter_size[-1] + (2 * padding)) / c_stride) + 1)
        o_shape = (o_rows, o_cols)

        return o_shape

    def forward(self, inputs: Tensor) -> Tensor:
        print("----------------------convolution forward  just started------------------------------------------------")
        # convolve

        self.inputs = inputs

        # output = self.convolve(self.inputs, self.filters, r_stride=self.r_stride, c_stride=self.c_stride)
        output = self.convolve(self.inputs, self.params["w"], r_stride=self.r_stride, c_stride=self.c_stride)
        # add biases
        for image in range(output.shape[0]):
            for chl in range(output.shape[1]):
                output[image][chl] += self.params["b"][chl]

        print("----------------------convolution forward  just finished-----------------------------------------------")
        return output

    def backward(self, grad: Tensor) -> Tensor:

        output_size = self.calc_output_size(np.shape(self.inputs), np.shape(grad), self.r_stride, self.c_stride)

        output_rows = output_size[0]  # ?
        output_cols = output_size[1]  # ?
        n_images = len(self.inputs)
        grad_rows = np.shape(grad)[-2]
        grad_cols = np.shape(grad)[-1]

        grad_w = np.zeros(np.shape(self.filters))

        filter_channel_idx = np.ndindex(np.shape(grad)[1:-2])
        input_channel_idx = np.ndindex(np.shape(self.inputs)[1:-2])
        for img in range(0, n_images):
            for f in filter_channel_idx:
                for c in input_channel_idx:
                    for row in range(0, output_rows, self.r_stride):
                        for col in range(0, output_cols, self.c_stride):

                            sub_inputs = self.inputs[img][c][row: row + grad_rows, col: col + grad_cols]

                            grad_w[f][c][row][col] = np.sum(np.multiply(sub_inputs, grad[img][f]))

        self.grads["w"] = grad_w

        grad_b = np.zeros(self.n_filters)
        channels = np.ndindex(np.shape(grad)[1:-2])

        for image in range(n_images):
            for chl in channels:

                grad_b[chl] += np.sum(grad[image][chl])

        self.grads["b"] = grad_b

        padding_rc = calculate_padding(np.shape(grad), np.shape(self.params["w"]),
                                       np.shape(self.inputs), self.r_stride, self.c_stride)

        p_grad = pad(grad, padding_rc)

        output_size = self.calc_output_size(np.shape(p_grad), np.shape(self.params["w"]), self.r_stride, self.c_stride)
        output_rows = output_size[0]  # ?
        output_cols = output_size[1]  # ?

        gradient = np.zeros((np.shape(self.inputs)))

        p_grad_channel_idx = np.ndindex(np.shape(p_grad)[1:-2])


        reshaped_filter = self.params["w"].reshape((np.shape(self.params["w"])[1], np.shape(self.params["w"])[0],
                                                            np.shape(self.params["w"])[-2], np.shape(self.params["w"])[-1]))

        inverted_filter = np.rot90(reshaped_filter, k=2, axes=(-2, -1))


        for img in range(0, n_images):
            for f in range(0, len(reshaped_filter)):
                for c in p_grad_channel_idx:
                    for row in range(0, output_rows, self.r_stride):
                        for col in range(0, output_cols, self.c_stride):

                            sub_inputs = p_grad[img][c][row: row + self.f_rows, col: col + self.f_cols]

                            gradient[img][f][row][col] = np.sum(np.multiply(sub_inputs, inverted_filter[f][c]))


        print("----------------------convolution backward just finished-----------------------------------------------")
        return gradient


    def convolve(self, inputs: Tensor, filters: Sequence[Tensor], r_stride: int = 1, c_stride: int = 1):

        # ---------------------------------------TEST VERSION-------------------------------------------------------
        output_size = self.calc_output_size(np.shape(inputs), np.shape(filters), r_stride, c_stride)

        output_rows = output_size[0]  # ?
        output_cols = output_size[1]  # ?
        n_images = len(inputs)
        f_rows = np.shape(filters)[-2]
        f_cols = np.shape(filters)[-1]
        n_filters = len(filters)

        output = np.zeros((n_images, n_filters, output_rows, output_cols))

        output_index = (index[0] for index in np.ndenumerate(inputs[..., ::r_stride, ::c_stride])
                            if index[0][-1] <= output_cols - 1 and index[0][-2] <= output_rows - 1)

        for idx in output_index:

            for f in range(n_filters):
                input_row = r_stride * idx[-2]
                input_col = c_stride * idx[-1]
                image = idx[0]

                sub_inputs = inputs[idx[0:-2]][input_row: input_row + f_rows, input_col: input_col + f_cols]

                output[image][f][idx[-2]][idx[-1]] += np.sum(np.multiply(sub_inputs, filters[f][idx[1:-2]]))

        return output


class Normalization(Layer):

    def __init__(self, gamma=1, beta=1, epsilon=1e-5):

        super().__init__()

        self.gamma = gamma
        self.beta = beta
        self.params["g"] = np.array([''])
        self.params["B"] = np.array([''])
        self.epsilon = epsilon
        self.x_hat = np.array([''])
        self.batch_var = np.array([''])
        self.batch_size = len(self.inputs)
        self.params["g"] = self.gamma * np.ones(np.shape(self.inputs[0]))
        self.params["B"] = self.beta * np.ones(np.shape(self.inputs[0]))


    def forward(self, inputs: Tensor) -> Tensor:

        self.inputs = inputs
        batch_mean = self.inputs.mean(axis=0)
        batch_var = self.inputs.var(axis=0)
        batch_std = np.sqrt(batch_var + self.epsilon)

        inputs_centered = self.inputs - batch_mean

        self.x_hat = inputs_centered / batch_std

        output = self.gamma * self.x_hat + self.beta

        return output

    def backward(self, grad: Tensor) -> Tensor:

        #DelL/Del(Gamma)

        self.grads["g"] = np.sum(grad, axis=0) * self.x_hat

        # DelL/Del(Beta)

        self.grads["B"] = np.sum(grad, axis=0)

        # DelL/Del(inputs_normalized)

        d_x_hat = grad * self.gamma

        # DelL/Del(inputs)

        sqrt_inv_var = 1/np.sqrt(self.batch_var + self.epsilon)

        gradient = (1/self.batch_size) * sqrt_inv_var * (self.batch_size*d_x_hat - np.sum(d_x_hat, axis=0)
                                                         - self.x_hat * np.sum(d_x_hat*self.x_hat, axis=0))

        return gradient


def max_pool(sub_inputs: Tensor) ->int:
    return np.amax(sub_inputs)


def max_pool_idx(sub_inputs: Tensor) -> Tensor:
    return np.argmax(sub_inputs)


def average_pool(sub_input: Tensor) -> int:
    return np.average(sub_input)


class Pooling(Layer):

    def __init__(self, input_size: tuple[int, ...],
                 filter_size: tuple[int, ...],
                 r_stride: int,
                 c_stride: int,
                 pool_type: str = "max",
                 padding: int = 0,
                 flatten_output=False

                 ) -> None:
        super().__init__()
        self.input_size = input_size
        self.f_rows = filter_size[-2]
        self.f_cols = filter_size[-1]
        self.channels = input_size[-3]
        self.r_stride = r_stride
        self.c_stride = c_stride
        self.padding = padding
        self.pool_type = pool_type
        self.flatten_output = flatten_output
        self.output_size = self.calc_output_size(input_size, (self.f_rows, self.f_cols), self.r_stride, self.c_stride,
                                                 self.padding)

    def calc_output_size(self, input_size: tuple[int, ...], filter_size: tuple[int, ...], r_stride: int,
                         c_stride: int, padding: int):

        o_rows: int = int(((input_size[-2] - filter_size[-2] + (2 * padding)) / r_stride) + 1)
        o_cols: int = int(((input_size[--1] - filter_size[-1] + (2 * padding)) / c_stride) + 1)

        return o_rows, o_cols

    def pooling_type(self, sub_inputs: Tensor, pool_type: str = 'max') -> int:

        match pool_type:
            case 'max':
                return max_pool(sub_inputs)
            case 'average':
                return average_pool(sub_inputs)
            case _:
                raise Exception(""" pooling_type must take the value: "max' or 'average' """)

    def pool(self, inputs: Tensor) -> Tensor:
        print("------------------------------------pool just started-----------------------------------------------")
        output_rows = self.output_size[-2]
        output_cols = self.output_size[-1]
        n_images = len(inputs)
        output = np.zeros((n_images, self.channels, output_rows, output_cols))
        output_rc = (index[0] for index in np.ndenumerate(inputs[..., ::self.r_stride, ::self.c_stride])
                     if index[0][-1] <= output_cols-1 and index[0][-2] <= output_rows-1)


        for index in output_rc:

            input_row = self.r_stride * index[-2]
            input_col = self.c_stride * index[-1]
            channel = index[1:-2]
            image = index[0]

            sub_inputs = inputs[image][channel][input_row: input_row + self.f_rows, input_col: input_col + self.f_cols]

            output[image][channel][index[-2]][index[-1]] = self.pooling_type(sub_inputs, self.pool_type)


        if self.flatten_output:
            output = output.reshape(n_images, self.channels * output_rows * output_cols)

        print("----------------------pool just finished---------------------------------------------------------------")
        return output

    def forward(self, inputs: Tensor) -> Tensor:
        print("----------------------pool forward  just started-------------------------------------------------------")
        self.inputs = inputs

        print("----------------------pool forward  just finished------------------------------------------------------")
        return self.pool(self.inputs)

    def backward(self, grad: Tensor) -> Tensor:
        print("----------------------pool backward just started-------------------------------------------------------")
        output_rows = self.output_size[-2]
        output_cols = self.output_size[-1]
        grad = np.zeros(np.shape(self.inputs))

        grad_rc = (index[0] for index in np.ndenumerate(self.inputs[..., ::self.r_stride, ::self.c_stride])
                   if index[0][-1] <= output_cols - 1 and index[0][-2] <= output_rows - 1)

        for index in grad_rc:
            input_row = self.r_stride * index[-2]
            input_col = self.c_stride * index[-1]
            channel = index[1:-2]
            image = index[0]

            sub_inputs = self.inputs[image][channel][input_row: input_row + self.f_rows,
                         input_col: input_col + self.f_cols]

            if self.pool_type == "max":
                idx_max_flat = max_pool_idx(sub_inputs)
                idx_max_row = int(idx_max_flat/self.f_cols)
                idx_max_col = idx_max_flat - (idx_max_row * self.f_cols)
                grad[image][channel][input_row + idx_max_row][input_col + idx_max_col] = 1
            else:
                Exception(" Pooling layer back propagation is only available for max pooling . ")

        print("----------------------pool backward just finished------------------------------------------------------")
        return grad


def check_dims_params(input_size, output_size, r_stride, c_stride):
    # inputs must have more dimensions than outputs
    if len(output_size) != 2:
        print("error: output dimensions should be in the form: (number_rows, number of columns)")
        raise Exception("convolutional layer class could not be initialised")

    # delete edge case once padding has been introduced
    elif input_size[-1] < output_size[-1] or input_size[-2] < output_size[-2]:
        print("error: output rows and columns must be between 1 and the number of input columns and rows")
        raise Exception("convolutional layer class could not be initialised")

    elif r_stride > input_size[-2] or c_stride > input_size[-1]:
        print("error: values of r_stride and r_stride arguments should be less than or equal to...\n"
              " number of output rows/columns ")
        raise Exception("convolutional layer class could not be initialised")

    else:

        # check if output size used in conv net matches output size used in

        if input_size[-1] / c_stride < output_size[-1] or input_size[-2] / r_stride < output_size[-2]:
            print("warning: number of output rows exceeds number max number of strides that can be taken")
            print("Rows/columns exceeding the maximum number of strides will be excluded from the calculation")


class Flatten(Layer):

    def __init__(self, input_size: tuple[int, ...]):
        super().__init__()
        self.input_size = input_size

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs

        return flatten_inputs(self.inputs)

    def backward(self, grad: Tensor) -> Tensor:

        return grad.reshape((np.shape(self.inputs)))


def soft_max(inputs: Tensor) -> Tensor:

    reduced_inputs = inputs
    output = np.zeros(np.shape(inputs))

    for img_idx in range(len(inputs)):



        reduced_inputs[img_idx] = inputs[img_idx] - np.amax(inputs)

    t_j = np.exp(reduced_inputs)


    for idx in range(len(t_j)):

        output[idx] = t_j[idx] / sum(t_j[idx])

    return output


def soft_max_prime(inputs: Tensor) -> Tensor:

    softmax = soft_max(inputs)
    s = np.zeros((len(softmax), len(softmax[0]), len(softmax[0])))
    s_flat = np.zeros((len(softmax), len(softmax[0])))

    for image in range(len(softmax)):

        s_img = softmax[image].reshape(-1, 1)

        s[image] = np.diagflat(s_img) - np.dot(s_img, s_img.T)

        s_flat[image] = np.diagonal(s[image])

    return s_flat


class SoftMax(Activation):

    def __init__(self) -> None:
        super().__init__(soft_max, soft_max_prime)



