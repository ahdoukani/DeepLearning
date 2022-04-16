"""
I used data from Kaggle open source database for prices of groceries from Local Mongolian stores".
(prices_en_encoded.csv)

This nn predicts future grocery prices for a given input date

"""

import pandas as pd
import numpy as np

from training_gh import train
from neural_net_gh import NeuralNet
from Layers_gh import Linear, Tanh

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ------------------------------------------------------------------------------------------------------------------
    #  ========================Project 2) future price prediction (Local Mongolian stores)=============================
    # -----------------------------------------------------------------------------------------------------------------
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # ------------------------------------------------------------------------------------------------------------------

    # Preparing data into dataframe and extracting training/ evaluation inputs and targets
    #  creates a dftrain pandas dataframe and Read csv file into it.
    dftrain = pd.read_csv("prices_en_encoded.csv")
    # tilde is th bitwise negation operator- flips any binary number to obtain its complementary (e.g ~0  is 1)
    # looking for only "Altan Taria Flour, First Grade" products that don't have a null value
    dftrain = dftrain[(dftrain["product"] == "Altan Taria Flour, First Grade") & (~dftrain['price'].isnull())]
    # sort these values by ascending date
    dftrain.sort_values(by="date", ascending=True)
    #copy dftrain dataframe into dfeval variable
    dfeval = dftrain.copy()

    # select the first 6000 entries from columns: date, product encoded and price column for training data
    dftrain = dftrain.loc[1:6000, ["date", "product encoded", "price"]]
    # select the next 6000 entries from columns: date, product encoded and price column for evaluation data
    dfeval = dfeval.loc[6001:12000, ["date", "product encoded", "price"]]

    # input date
    t_inputs, e_inputs = dftrain.loc[:, "date"], dfeval.loc[:, "date"]
    # target outputs: price
    t_targets, e_targets = dftrain.loc[:, "price"], dfeval.loc[:, "price"]

    # convert these to numpy arrays for compatibility with the deep learning module
    t_inputs, t_targets = t_inputs.to_numpy(), t_targets.to_numpy()
    t_inputs, t_targets = np.reshape(t_inputs, (-1, 1)), np.reshape(t_targets, (-1, 1))
    e_inputs, e_targets = e_inputs.to_numpy(), e_targets.to_numpy()
    e_inputs, e_targets = np.reshape(e_inputs, (-1, 1)), np.reshape(e_targets, (-1, 1))

    print('training inputs:', t_inputs)
    print(' training targets:', t_targets)

    print('-----------------------------------------------------------------------------------------------------------')
    print('///////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('-----------------------------------1) CREATING NEURAL NETWORK----------------------------------------------')
    print('///////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('-----------------------------------------------------------------------------------------------------------')


    # project 2) estimating future grocery store prices (local mongolian stores)
    net = NeuralNet([
        Linear(input_size=1, output_size=1),
        Tanh(),
        Linear(input_size=1, output_size=1)

    ])

    print('-----------------------------------------------------------------------------------------------------------')
    print('///////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('-----------------------------------2) MODEL TRAINING PROCESS-----------------------------------------------')
    print('///////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('-----------------------------------------------------------------------------------------------------------')

    train(net, t_inputs, t_targets)

    # output--------------------------------------------------------------
    print('-----------------------------------------------------------------------------------------------------------')
    print('///////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('-----------------------------------3) OUTPUTTING RESULTS --------------------------------------------------')
    print('///////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('-----------------------------------------------------------------------------------------------------------')

    for x, y in zip(e_inputs, e_targets):
        print("shape of test_images (x) ", np.shape(x))
        print("shape of test_labels (y) ", np.shape(y))
        predicted = net.forward(x)
        print(f' predicted value: {predicted}, actual value (target value):  {y}')

