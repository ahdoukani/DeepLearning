"""
Optimizer uses computed gradients to adjust parameters,
... which are then used make predictions for the next batch of data
SGD Optimizer was included in this module from Joel Grus's  Git-hub and online tutorial
https://github.com/joelgrus/joelnet

I have extended this module with my own implementation of SGDM, RMSProp, and Adam Optimizers.
each class has its own print parameter method
"""

from neural_net_gh import NeuralNet
from tensor import Tensor
import numpy as np
from typing import Dict


class Optimizer:

    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

    def init_calculated_params(self, net: NeuralNet):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def init_calculated_params(self, net: NeuralNet):
        print(" SGD requires no calculated params")
        pass

    print(f' Optimizers.SGD() just initialised')

    def step(self, net: NeuralNet) -> None:
        print(f' Optimizers.step  started running')
        for param, grad in net.params_and_grads():


            param -= self.lr * grad
        print(f' Optimizers.step finished running')


class SGDM(Optimizer):
    def __init__(self, lr: float = 0.1, momentum: float = 0.9) -> None:

        self.lr = lr
        self.momentum = momentum
        self.param_velocity: Dict[str:Dict[str, Tensor]] = {}

        print(f' Optimizers.SGDM() just initialized')

    def init_calculated_params(self, net: NeuralNet):

        for i in range(0, len(net.layers)):

            print('layer:', i)

            if all(k in net.layers[i].params for k in ("w", "b")):
                self.param_velocity[str(i)] = {'w': np.zeros(net.layers[i].params['w'].shape),
                                               'b': np.zeros(net.layers[i].params['b'].shape)}

            if all(k in net.layers[i].params for k in ("g", "B")):
                self.param_velocity[str(i)] = {'w': np.zeros(net.layers[i].params['g'].shape),
                                               'b': np.zeros(net.layers[i].params['B'].shape)}

    def update_param_velocity(self, net: NeuralNet):

        for i in range(0, len(net.layers)):

            if all(k in net.layers[i].params for k in ("w", "b")):
                self.param_velocity[str(i)]['w'] = ((self.momentum * self.param_velocity[str(i)]['w']) +
                                                    ((1 - self.momentum) * net.layers[i].grads['w']))

                self.param_velocity[str(i)]['b'] = ((self.momentum * self.param_velocity[str(i)]['b']) +
                                                    ((1 - self.momentum) * net.layers[i].grads['b']))

            if all(k in net.layers[i].params for k in ("g", "B")):
                self.param_velocity[str(i)]['g'] = ((self.momentum * self.param_velocity[str(i)]['g']) +
                                                    ((1 - self.momentum) * net.layers[i].grads['g']))

                self.param_velocity[str(i)]['B'] = ((self.momentum * self.param_velocity[str(i)]['B']) +
                                                    ((1 - self.momentum) * net.layers[i].grads['B']))

    def print_params(self, net: NeuralNet):

        print(f' ---------------------SGDM: Printing Values-----------------------------------------------------------')
        for i in range(0, len(net.layers)):
            if all(k in net.layers[i].params for k in ("w", "b")):
                print(f'net.layers[{i}].grads[w]', net.layers[i].grads['w'])
                print(f'self.param_velocity[{i}][w] ', self.param_velocity[str(i)]['w'])
                print('CALCULATED NEW WEIGHTS FOR LAYER- net.layers[i].params[w]', net.layers[i].params["w"])
                print('---------------------------------------------------------')

                print(f'net.layers[{i}].grads[b]', net.layers[i].grads['b'])
                print(f'self.param_velocity[{i}][b] ', self.param_velocity[str(i)]['b'])
                print(f'CALCULATED NEW BIASES FOR LAYER - net.layers[{i}].params[b]', net.layers[i].params["b"])
                print('---------------------------------------------------------')

    def step(self, net: NeuralNet) -> None:

        print(f' ----------------------Optimizers.SGDM().step just just started calculating new params----------------')

        self.update_param_velocity(net)

        for i in range(0, len(net.layers)):
            if all(k in net.layers[i].params for k in ("w", "b")):

                net.layers[i].params['w'] -= self.lr * self.param_velocity[str(i)]["w"]
                net.layers[i].params['b'] -= self.lr * self.param_velocity[str(i)]["b"]

                if all(k in net.layers[i].params for k in ("g", "B")):
                    net.layers[i].params['g'] -= self.lr * self.param_velocity[str(i)]["g"]
                    net.layers[i].params['B'] -= self.lr * self.param_velocity[str(i)]["B"]

        # self.print_params(net)

        print(f' -----------------Optimizers.SGDM().step finished running-------------------------')


class RMSProp(Optimizer):

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        self.lr = lr
        self.beta = beta
        self.param_speed: Dict[str:Dict[str, Tensor]] = {}
        self.epsilon = 0.00000001  # as is recommended to use in keras module

        print(f' Optimizers.RMSProp() just initialized')

    def init_calculated_params(self, net: NeuralNet):
        for i in range(0, len(net.layers)):



            if all(k in net.layers[i].params for k in ("w", "b")):
                self.param_speed[str(i)] = {'w': np.zeros(net.layers[i].params['w'].shape),
                                            'b': np.zeros(net.layers[i].params['b'].shape)}

            if all(k in net.layers[i].params for k in ("g", "B")):
                self.param_speed[str(i)] = {'g': np.zeros(net.layers[i].params['g'].shape),
                                            'B': np.zeros(net.layers[i].params['B'].shape)}

    def update_param_speed(self, net: NeuralNet):

        for i in range(0, len(net.layers)):



            if all(k in net.layers[i].params for k in ("w", "b")):
                self.param_speed[str(i)]['w'] = (self.beta * self.param_speed[str(i)]['w'] +
                                                 ((1 - self.beta) * np.square(net.layers[i].grads['w'])))
                self.param_speed[str(i)]['b'] = (self.beta * self.param_speed[str(i)]['b'] +
                                                 ((1 - self.beta) * np.square(net.layers[i].grads['b'])))

                self.param_speed[str(i)]['g'] = (self.beta * self.param_speed[str(i)]['g'] +
                                                 ((1 - self.beta) * np.square(net.layers[i].grads['g'])))
                self.param_speed[str(i)]['B'] = (self.beta * self.param_speed[str(i)]['B'] +
                                                 ((1 - self.beta) * np.square(net.layers[i].grads['B'])))

    def print_params(self, net: NeuralNet):
        print(f' ------------------------------RMSProp: printing params-----------------------------------------------')
        for i in range(0, len(net.layers)):
            if all(k in net.layers[i].params for k in ("w", "b")):
                print('net.layers[i].grads[w]', net.layers[i].grads['w'])
                print('self.param_speed[i][w] ', self.param_speed[str(i)]['w'])
                print('CALCULATED NEW WEIGHTS FOR LAYER- net.layers[i].params[w]', net.layers[i].params['w'])
                print('---------------------------------------------------------')
                print('self.param_speed[i][b] ', self.param_speed[str(i)]['b'])
                print('net.layers[i].grads[b]', net.layers[i].grads['b'])
                print('CALCULATED NEW BIASES FOR LAYER - net.layers[i].params[b]', net.layers[i].params['b'])

            if all(k in net.layers[i].params for k in ("g", "B")):
                print('net.layers[i].grads[g]', net.layers[i].grads['g'])
                print('self.param_speed[i][g] ', self.param_speed[str(i)]['g'])
                print('CALCULATED NEW WEIGHTS FOR LAYER- net.layers[i].params[g]', net.layers[i].params['g'])
                print('---------------------------------------------------------')
                print('self.param_speed[i][B] ', self.param_speed[str(i)]['B'])
                print('net.layers[i].grads[B]', net.layers[i].grads['B'])
                print('CALCULATED NEW BIASES FOR LAYER - net.layers[i].params[B]', net.layers[i].params['B'])

    def step(self, net: NeuralNet) -> None:

        self.update_param_speed(net)

        print(f' -----------------Optimizers.RMSProp().step just just started calculating new params-----------')

        for i in range(0, len(net.layers)):
            if all(k in net.layers[i].params for k in ("w", "b")):
                net.layers[i].params['w'] -= self.lr * np.divide(net.layers[i].grads['w'],
                                                                 np.sqrt(self.param_speed[str(i)]['w']) + self.epsilon)
                net.layers[i].params['b'] -= self.lr * np.divide(net.layers[i].grads['b'],
                                                                 np.sqrt(self.param_speed[str(i)]['b']) + self.epsilon)

            if all(k in net.layers[i].params for k in ("g", "B")):
                net.layers[i].params['g'] -= self.lr * np.divide(net.layers[i].grads['g'],
                                                                 np.sqrt(self.param_speed[str(i)]['g']) + self.epsilon)
                net.layers[i].params['B'] -= self.lr * np.divide(net.layers[i].grads['B'],
                                                                 np.sqrt(self.param_speed[str(i)]['B']) + self.epsilon)

        print(f' -----------------Optimizers.RMSProp().step finished calculations-------------------------')
        self.print_params(net)
        print(f' -----------------Optimizers.RMSProp().step finished running-------------------------------')


class Adam(Optimizer):

    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999) -> None:

        self.lr = lr
        self.beta1 = beta1  # momentum
        self.beta2 = beta2  # beta rmsprop
        self.param_speed: Dict[str:Dict[str, Tensor]] = {}
        self.param_velocity: Dict[str:Dict[str, Tensor]] = {}
        self.epsilon = 0.00000001  # as set by keras module
        self.corrected_speed: Dict[str:Dict[str, Tensor]] = {}
        self.corrected_velocity: Dict[str:Dict[str, Tensor]] = {}
        self.t: int = 1

        print(f' Optimizers.Adam() just initialized')

    def init_calculated_params(self, net: NeuralNet):
        for i in range(0, len(net.layers)):


            if all(k in net.layers[i].params for k in ("w", "b")):
                self.param_speed[str(i)] = {'w': np.zeros(net.layers[i].params['w'].shape),
                                            'b': np.zeros(net.layers[i].params['b'].shape)}

                self.param_velocity[str(i)] = {'w': np.zeros(net.layers[i].params['w'].shape),
                                               'b': np.zeros(net.layers[i].params['b'].shape)}
                self.corrected_velocity[str(i)] = {'w': np.zeros(net.layers[i].params['w'].shape),
                                                   'b': np.zeros(net.layers[i].params['b'].shape)}
                self.corrected_speed[str(i)] = {'w': np.zeros(net.layers[i].params['w'].shape),
                                                'b': np.zeros(net.layers[i].params['b'].shape)}

            if all(k in net.layers[i].params for k in ("g", "B")):
                self.param_speed[str(i)] = {'g': np.zeros(net.layers[i].params['g'].shape),
                                            'B': np.zeros(net.layers[i].params['b'].shape)}

                self.param_velocity[str(i)] = {'g': np.zeros(net.layers[i].params['g'].shape),
                                               'B': np.zeros(net.layers[i].params['B'].shape)}
                self.corrected_velocity[str(i)] = {'g': np.zeros(net.layers[i].params['g'].shape),
                                                   'B': np.zeros(net.layers[i].params['B'].shape)}
                self.corrected_speed[str(i)] = {'g': np.zeros(net.layers[i].params['g'].shape),
                                                'B': np.zeros(net.layers[i].params['B'].shape)}

    def update_param_adam(self, net: NeuralNet):

        for i in range(0, len(net.layers)):


            if all(k in net.layers[i].params for k in ("w", "b")):
                self.param_velocity[str(i)]['w'] = ((self.beta1 * self.param_velocity[str(i)]['w']) +
                                                    ((1 - self.beta1) * net.layers[i].grads['w']))

                self.param_velocity[str(i)]['b'] = ((self.beta1 * self.param_velocity[str(i)]['b']) +
                                                    ((1 - self.beta1) * net.layers[i].grads['b']))

                self.param_speed[str(i)]['w'] = (self.beta2 * self.param_speed[str(i)]['w'] +
                                                 ((1 - self.beta2) * np.square(net.layers[i].grads['w'])))
                self.param_speed[str(i)]['b'] = (self.beta2 * self.param_speed[str(i)]['b'] +
                                                 ((1 - self.beta2) * np.square(net.layers[i].grads['b'])))

                self.corrected_speed[str(i)]['w'] = np.divide(self.param_speed[str(i)]['w'], 1 - (self.beta2 ** self.t))
                self.corrected_speed[str(i)]['b'] = np.divide(self.param_speed[str(i)]['b'], 1 - (self.beta2 ** self.t))

                self.corrected_velocity[str(i)]['w'] = np.divide(self.param_velocity[str(i)]['w'],
                                                                 1 - (self.beta1 ** self.t))
                self.corrected_velocity[str(i)]['b'] = np.divide(self.param_velocity[str(i)]['b'],
                                                                 1 - (self.beta1 ** self.t))

            if all(k in net.layers[i].params for k in ("g", "B")):
                self.param_velocity[str(i)]['g'] = ((self.beta1 * self.param_velocity[str(i)]['g']) +
                                                    ((1 - self.beta1) * net.layers[i].grads['g']))

                self.param_velocity[str(i)]['B'] = ((self.beta1 * self.param_velocity[str(i)]['B']) +
                                                    ((1 - self.beta1) * net.layers[i].grads['B']))

                self.param_speed[str(i)]['g'] = (self.beta2 * self.param_speed[str(i)]['g'] +
                                                 ((1 - self.beta2) * np.square(net.layers[i].grads['g'])))
                self.param_speed[str(i)]['B'] = (self.beta2 * self.param_speed[str(i)]['B'] +
                                                 ((1 - self.beta2) * np.square(net.layers[i].grads['B'])))

                self.corrected_speed[str(i)]['g'] = np.divide(self.param_speed[str(i)]['g'], 1 - (self.beta2 ** self.t))
                self.corrected_speed[str(i)]['B'] = np.divide(self.param_speed[str(i)]['B'], 1 - (self.beta2 ** self.t))

                self.corrected_velocity[str(i)]['g'] = np.divide(self.param_velocity[str(i)]['g'],
                                                                 1 - (self.beta1 ** self.t))
                self.corrected_velocity[str(i)]['B'] = np.divide(self.param_velocity[str(i)]['B'],
                                                                 1 - (self.beta1 ** self.t))

    def print_params(self, net: NeuralNet):
        print(f' ------------------------------Adam(): printing params-----------------------------------------------')
        for i in range(0, len(net.layers)):
            if all(k in net.layers[i].params for k in ("w", "b")):
                print('net.layers[i].grads[w]', net.layers[i].grads['w'])
                print('self.param_speed[i][w] ', self.param_speed[str(i)]['w'])
                print('CALCULATED NEW WEIGHTS FOR LAYER- net.layers[i].params[w]', net.layers[i].params['w'])
                print('---------------------------------------------------------')
                print('self.param_speed[i][b] ', self.param_speed[str(i)]['b'])
                print('net.layers[i].grads[b]', net.layers[i].grads['b'])
                print('CALCULATED NEW BIASES FOR LAYER - net.layers[i].params[b]', net.layers[i].params['b'])

    def step(self, net: NeuralNet) -> None:

        self.update_param_adam(net)

        print(f' -----------------Optimizers.Adam().step just just started calculating new params-----------')

        for i in range(0, len(net.layers)):
            if all(k in net.layers[i].params for k in ("w", "b")):
                x_w = self.corrected_velocity[str(i)]['w']
                y_w = np.sqrt(self.corrected_speed[str(i)]['w']) + self.epsilon
                net.layers[i].params['w'] -= self.lr * np.divide(x_w, y_w)
                x_b = self.corrected_velocity[str(i)]['b']
                y_b = np.sqrt(self.corrected_speed[str(i)]['b']) + self.epsilon
                net.layers[i].params['b'] -= self.lr * np.divide(x_b, y_b)

            if all(k in net.layers[i].params for k in ("g", "B")):
                x_w = self.corrected_velocity[str(i)]['g']
                y_w = np.sqrt(self.corrected_speed[str(i)]['g']) + self.epsilon
                net.layers[i].params['g'] -= self.lr * np.divide(x_w, y_w)
                x_b = self.corrected_velocity[str(i)]['w']
                y_b = np.sqrt(self.corrected_speed[str(i)]['w']) + self.epsilon
                net.layers[i].params['w'] -= self.lr * np.divide(x_b, y_b)

        self.t += 1

        print(f' -----------------Optimizers.Adam().step finished calculations-------------------------')
        # self.print_params(net)
        print(f' -----------------Optimizers.Adam().step finished running-------------------------------')
