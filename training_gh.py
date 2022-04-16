"""
This function is used to train the neural net-
training is composed of
1) forward propagation
2) calculating epoch loss- represents cumilative loss after one iteration through the data-set
    - this is done with a suitable loss function
3) backward propagation- finding gradients of loss function and w.r.t trainable parameters
4) adjust parameters according to optimization function and its hyper parameters

This code is adapted from Joel Grus's Git hub that was instrumental getting me started with Neural Nets.
https://github.com/joelgrus/joelnet
Some variable changes have been made no additions were made except comments.

I have added comments and print statments to better understand what the code is doing and track training progress
I have added a function to initalize Hyper parameters to accommodate for the optimizers i wrote
( all optimizers except SGD)
"""


from tensor import Tensor
from neural_net_gh import NeuralNet
from Loss_function_gh import Loss, MSE, CrossEntropy
from optimizers_gh import Optimizer, SGD, SGDM, RMSProp, Adam
from data_gh import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 15,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = CrossEntropy(),
          optimizer: Optimizer = Adam()) -> None:

    print(f' --------------------training.train begins running------------')
    # initialising hyper parameters for optimization functions.
    optimizer.init_calculated_params(net)
    for epoch in range(num_epochs):
        print("EPOCH: ", epoch)
        # initializing epoch loss to 0 to ensure epoch losses are independent of each other
        epoch_loss = 0.0
        batch_idx = 0
        for batch in iterator(inputs, targets):
            print("BATCH: ", batch_idx)
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
            batch_idx += 1
        print("epoch:", epoch, "epoch loss: ", epoch_loss)


