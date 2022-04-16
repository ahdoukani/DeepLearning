
from typing import Sequence, Iterator, Tuple

from tensor import Tensor
from layers import Layer


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
        print('-----------------------------------NEURAL NETWORK CREATED-----------------------------------')

    def forward(self, inputs: Tensor) -> Tensor:
        print(f' this is neuralNet.forward just ran')
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            # print(f' this is neuralNet.backwards just ran and returned : {grad}')
            # print(f' this is neuralNet.backwards  we compare {reversed(self.layers)} and {grad}')
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        print("running params and grads")
        for layer in self.layers:
            # print('new layer')
            for name, param in layer.params.items():
                grad = layer.grads[name]
                # print('param: ', param)
                # print('grad: ', param)

                yield param, grad
