
"""
Loss functions will calculate the losses which indicate
how much are predictions depart from the target( actual) values.

The base class "Loss" and "MSE" was implemented by Joel Grus https://github.com/joelgrus/joelnet

I have extended functionality by writing my own implementation of  CrossEntropy loss function.
This loss function is well suited to feature classification type problems.


"""
import numpy as np

from tensor import Tensor


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    Calculates mean squared error (loss) and gradient of mean squared error (grad)
    """

    # print(f'sum( ((input to layer)*w + b) -( target))^2 = sum((predicted-target)^2)')
    # print(np.sum((predicted - actual) ** 2))
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        print(f' MSE(Loss).loss function used-  total loss calculated -> ')

        return np.sum((predicted - actual) ** 2)  # type: ignore

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        print(f' MSE(Loss).grad function used- grad of loss calculated')
        print(f' MSE(Loss).loss function used-  grads of loss function calculated -> ')
        # print(f' 2((input to layer)*w + b -( target)) = 2(predicted-target)')
        # print(2 * (predicted - actual))
        return 2 * (predicted - actual)


class CrossEntropy(Loss):

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        print(f' CrossEntropy(Loss).loss function used-  total loss calculated -> ')
        predicted = predicted
        print("predicted:  {}   actual: {}", np.shape(predicted), np.shape(actual))
        return -np.sum(actual * np.log(predicted))/len(predicted)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        print(f' CrossEntropy.grad function used- grad of loss calculated')

        return predicted - actual
