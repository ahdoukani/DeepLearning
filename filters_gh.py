"""
I have written functions that create some common edge detectors of specified number, ND channels

"""

import numpy as np
import tensor as Tensor
from typing import Union


def create_zeros_array(shape: tuple[int]):
    return np.zeros(shape)


def std_hv(filter_shape: tuple[int, ...]
           , zero_channels: Union[tuple[int, ...], tuple[tuple[int, ...], ...]] = ()
           , feature="horizontal"
           ) -> Tensor:
    #  zero_channels tuple is a tuple of indexes for input channels in you don't want to apply the filter to.
    #  comprehension : output channel for each channel in 'f' if channel is not in zero_channels tuple.

    f = create_zeros_array(filter_shape)

    non_zero_channels = (channel[0] for channel in np.ndenumerate(f) if channel not in zero_channels)

    if feature == "horizontal":
        if filter_shape[-2] >= 2:
            for idx_chl_2d in non_zero_channels:
                f[idx_chl_2d[0:-2]][0:int(np.floor(filter_shape[-2] / 2)), :] = 1
                f[idx_chl_2d[0:-2]][int(np.floor(filter_shape[-2] / 2)), :] = 0
                f[idx_chl_2d[0:-2]][int(np.ceil(filter_shape[-2] / 2)): filter_shape[-2], :] = -1
        else:
            print(" filter not created")
            raise Exception(" filter needs to have at least 2 rows for horizontal edge detection")

    elif feature == "vertical":
        if filter_shape[-1] >= 2:
            for idx_chl_2d in non_zero_channels:
                f[idx_chl_2d[0:-2]][:, 0:int(np.floor(filter_shape[-1] / 2))] = 1
                f[idx_chl_2d[0:-2]][:, int(np.floor(filter_shape[-1] / 2))] = 0
                f[idx_chl_2d[0:-2]][:, int(np.ceil(filter_shape[-1] / 2)): filter_shape[-1]] = -1
        else:
            print(" filter not created")
            raise Exception(" filter needs to have at least 2 columns for vertical edge detection")
    else:
        raise Exception(" invalid value for argument 3 (feature). Argument 3 is one of 'horizontal' or 'vertical'.")

    return f


def sobel_hv(filter_shape: tuple[int, ...]
             , zero_channels: Union[tuple[int, ...], tuple[tuple[int, ...], ...]] = ()
             , feature="horizontal"
             ) -> Tensor:
    f = create_zeros_array(filter_shape)

    #  zero_channels tuple is a tuple of indexes for input channels in you don't want to apply the filter to.
    #  comprehension : output channel for each channel in 'f' if channel is not in zero_channels tuple.

    non_zero_channels = (channel for channel in np.ndindex(filter_shape[-2]) if channel not in zero_channels)

    if feature == "horizontal":
        if filter_shape[-2] >= 2:
            for idx_chl_2d in non_zero_channels:
                f[idx_chl_2d][0:int(np.floor(filter_shape[-2] / 2)), :] = 1
                f[idx_chl_2d][int(np.floor(filter_shape[-2] / 2)), :] = 0
                f[idx_chl_2d][int(np.ceil(filter_shape[-2] / 2)): filter_shape[-2], :] = -1
                f[idx_chl_2d][:, int(np.floor(filter_shape[-2] / 2))] = f[idx_chl_2d][:,
                                                                        int(np.floor(filter_shape[-1] / 2))] * 2
        else:
            print(" filter not created")
            raise Exception(" filter needs to have at least 2 rows for horizontal edge detection")

    elif feature == "vertical":
        if filter_shape[-1] >= 2:
            for idx_chl_2d in non_zero_channels:
                f[idx_chl_2d][:, 0:int(np.floor(filter_shape[-1] / 2))] = 1
                f[idx_chl_2d][:, int(np.floor(filter_shape[-1] / 2))] = 0
                f[idx_chl_2d][:, int(np.ceil(filter_shape[-1] / 2)): filter_shape[-1]] = -1
                f[idx_chl_2d][int(np.floor(filter_shape[-2] / 2)), :] = f[idx_chl_2d][
                                                                        int(np.floor(filter_shape[-2] / 2)), :] * 2
        else:
            print(" filter not created")
            raise Exception(" filter needs to have at least 2 columns for vertical edge detection")

    else:
        raise Exception(" invalid value for argument 3 (feature). Argument 3 is one of 'horizontal' or 'vertical'.")

    return f


def schorr_hv(filter_shape: tuple[int, ...]
              , zero_channels: Union[tuple[int, ...], tuple[tuple[int, ...], ...]] = ()
              , feature="horizontal"
              ) -> Tensor:
    f = create_zeros_array(filter_shape)
    #  zero_channels tuple is a tuple of indexes for input channels in you don't want to apply the filter to.
    #  comprehension : output channel for each channel in 'f' if channel is not in zero_channels tuple.

    non_zero_channels = (channel for channel in np.ndindex(filter_shape[-2]) if channel not in zero_channels)

    if feature == "horizontal":
        if filter_shape[-2] >= 2:
            for idx_chl_2d in non_zero_channels:
                f[idx_chl_2d][0:np.floor(filter_shape[-2] / 2), :] = 3
                f[idx_chl_2d][np.ceil(filter_shape[-2] / 2): filter_shape[-2], :] = -3
                f[idx_chl_2d][:, np.floor(filter_shape[-1] / 2)] = 10
                f[idx_chl_2d][np.floor(filter_shape[-2] / 2), :] = 0
        else:
            print(" filter not created")
            raise Exception(" filter needs to have at least 2 rows for horizontal edge detection")

    elif feature == "vertical":
        if filter_shape[-1] >= 2:
            for idx_chl_2d in non_zero_channels:
                f[idx_chl_2d][:, 0:int(np.floor(filter_shape[-1] / 2))] = 3
                f[idx_chl_2d][:, int(np.ceil(filter_shape[-1] / 2)): filter_shape[-1]] = -3
                f[idx_chl_2d][int(np.floor(filter_shape[-2] / 2)), :] = 10
                f[idx_chl_2d][:, int(np.floor(filter_shape[-1] / 2))] = 0
        else:
            print(" filter not created")
            raise Exception(" filter needs to have at least 2 columns for vertical edge detection")

    else:
        raise Exception(" invalid value for argument 3 (feature). Argument 3 is one of 'horizontal' or 'vertical'.")
    return f


