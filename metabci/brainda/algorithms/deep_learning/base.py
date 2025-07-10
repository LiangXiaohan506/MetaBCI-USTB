# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/23
# License: MIT License
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skorch.classifier import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, EpochScoring, Checkpoint, EarlyStopping


def compute_out_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    return int(
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def compute_same_pad1d(input_size, kernel_size, stride=1, dilation=1):
    all_padding = (stride - 1) * input_size - stride + dilation * (kernel_size - 1) + 1
    return (all_padding // 2, all_padding - all_padding // 2)


def compute_same_pad2d(input_size, kernel_size, stride=(1, 1), dilation=(1, 1)):
    ud = compute_same_pad1d(
        input_size[0], kernel_size[0], stride=stride[0], dilation=dilation[0]
    )
    lr = compute_same_pad1d(
        input_size[1], kernel_size[1], stride=stride[1], dilation=dilation[1]
    )
    return [*lr, *ud]


class MaxNormConstraintConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm_value=1, norm_axis=2, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(
                torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True)
            )
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= desired / norms
        return w


class MaxNormConstraintLinear(nn.Linear):
    def __init__(self, *args, max_norm_value=1, norm_axis=0, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            # similar behavior as keras MaxNorm constraint
            norms = torch.sqrt(
                torch.sum(torch.square(w), dim=self.norm_axis, keepdim=True)
            )
            desired = torch.clamp(norms, 0, self.max_norm_value)
            # instead of desired/(eps+norm), without changing norm in range
            w *= desired / norms
        return w


def _adabn_pre_forward_hook(self, inputs):
    old_training_state = self.training
    self.eval()
    # global AdaBN
    with torch.no_grad():
        if not hasattr(self, "num_samples_tracked"):
            self.num_samples_tracked = 0
            self.running_mean.data.zero_()
            self.running_var.data.zero_()
            self.running_var.data.fill_(1)
        k = len(inputs[0])
        self.num_samples_tracked += k

        module_name = self.__class__.__name__
        if "BatchNorm1d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2))
            var = torch.var(input[0], dim=(0, 2))
        elif "BatchNorm2d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3))
            var = torch.var(inputs[0], dim=(0, 2, 3))
        elif "BatchNorm3d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3, 4))
            var = torch.var(inputs[0], dim=(0, 2, 3, 4))

        # see https://www.sciencedirect.com/science/article/abs/pii/S003132031830092X
        d = mean - self.running_mean.data
        self.running_mean.data.add_(d * k / self.num_samples_tracked)
        self.running_var.data.add_(
            (var - self.running_var.data) * k / self.num_samples_tracked
            + torch.square(d)
            * k
            * (self.num_samples_tracked - k)
            / (self.num_samples_tracked**2)
        )

    if old_training_state:
        self.train()


def _global_adabn_pre_forward_hook(self, inputs):
    old_training_state = self.training
    self.eval()
    # global AdaBN
    with torch.no_grad():
        module_name = self.__class__.__name__
        if "BatchNorm1d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2))
            var = torch.var(inputs[0], dim=(0, 2))
        elif "BatchNorm2d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3))
            var = torch.var(inputs[0], dim=(0, 2, 3))
        elif "BatchNorm3d" in module_name:
            mean = torch.mean(inputs[0], dim=(0, 2, 3, 4))
            var = torch.var(inputs[0], dim=(0, 2, 3, 4))

        self.running_mean.data.zero_()
        self.running_mean.data.add_(mean)
        self.running_var.data.zero_()
        self.running_var.data.add_(var)
    if old_training_state:
        self.train()


def adaptive_batch_norm(model, use_global=False):
    # register pre_forward_hook
    handles = []
    hook = _global_adabn_pre_forward_hook if use_global else _adabn_pre_forward_hook
    for module in model.modules():
        if "BatchNorm" in module.__class__.__name__:
            handles.append(module.register_forward_pre_hook(hook))
    return model, handles


def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot uniform/xavier initialization, and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def _narrow_normal_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    narrow normal distribution N(0, 0.01).
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.normal_(module.weight, mean=0.0, std=1e-2)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class NeuralNetClassifierNoLog(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        return super(NeuralNetClassifier, self).get_loss(
            y_pred, y_true, *args, **kwargs
        )

    def fit(self, X, y, **fit_params):
        net = super(NeuralNetClassifier, self).fit(X, y, **fit_params)
        callbacks = OrderedDict(net.callbacks)
        if "checkpoint" in callbacks:
            net.load_params(checkpoint=callbacks["checkpoint"])
        return net

    ### ==============================添加内容=============================== ###
    def forward_(self, x):
        # convert to torch tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype != torch.float64:
            x = x.to(dtype=torch.float64)
        elif not isinstance(x, torch.Tensor):
            raise ValueError("Input X must be a numpy array or torch tensor.")
        return self.module.forward(x)
    ### ==============================添加内容=============================== ###


class SkorchNet:
    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        model = self.module(*args, **kwargs)
        net = NeuralNetClassifierNoLog(
            module = model.double(),
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            optimizer__weight_decay=0,
            batch_size=128,
            lr=1e-2,
            max_epochs=300,
            device="cpu",
            train_split=ValidSplit(0.2, stratified=True),
            iterator_train__shuffle=True,
            callbacks=[
                (
                    "train_acc",
                    EpochScoring(
                        "accuracy",
                        name="train_acc",
                        on_train=True,
                        lower_is_better=False,
                    ),
                ),
                ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=300 - 1)),
                ("estoper", EarlyStopping(patience=50)),
                (
                    "checkpoint",
                    Checkpoint(dirname="checkpoints/{:s}".format(str(id(model)))),
                ),
            ],
            verbose=True,
        )
        return net


def np_to_th(X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
    """
    Convenience function to transform numpy array to `torch.Tensor`.

    Converts `X` to ndarray using asarray if necessary.

    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor

    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor


class AvgPool2dWithConv(nn.Module):
    """
    Compute average pooling using a convolution, to have the dilation parameter.

    Parameters
    ----------
    kernel_size: (int,int)
        Size of the pooling region.
    stride: (int,int)
        Stride of the pooling operation.
    dilation: int or (int,int)
        Dilation applied to the pooling filter.
    padding: int or (int,int)
        Padding applied before the pooling operation.
    """

    def __init__(self, kernel_size, stride, dilation=1, padding=0):
        super(AvgPool2dWithConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        # don't name them "weights" to
        # make sure these are not accidentally used by some procedure
        # that initializes parameters or something
        self._pool_weights = None

    def forward(self, x):
        # Create weights for the convolution on demand:
        # size or type of x changed...
        in_channels = x.size()[1]
        weight_shape = (
            in_channels,
            1,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        if self._pool_weights is None or (
            (tuple(self._pool_weights.size()) != tuple(weight_shape))
            or (self._pool_weights.is_cuda != x.is_cuda)
            or (self._pool_weights.data.type() != x.data.type())
        ):
            n_pool = np.prod(self.kernel_size)
            weights = np_to_th(np.ones(weight_shape, dtype=np.float32) / float(n_pool))
            weights = weights.type_as(x)
            if x.is_cuda:
                weights = weights.cuda()
            self._pool_weights = weights

        pooled = F.conv2d(
            x,
            self._pool_weights,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=in_channels,
        )
        return pooled


class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return self.__class__.__name__ + "(expression=%s) " % expression_str


class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


def identity(x):
    return x


def transpose_time_to_spat(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0, 3, 2, 1)


def squeeze_final_output(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x



### ==============================添加内容=============================== ###
class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)
        if self.bias:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        # if self.bias:
        #     self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

    def __call__(self, *input, **kwargs):
        return super()._call_impl(*input, **kwargs)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bias=False, WeightNorm=False, group=False, max_norm=1.):
        super(TemporalBlock, self).__init__()
        if group:
            if n_inputs >= n_outputs:
                self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                                  dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_outputs)
            else:
                self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                                  dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_inputs)
            self.conv1_point = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu1 = nn.ELU() # inplace=True
        self.dropout1 = nn.Dropout(dropout)

        if group:
            self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_outputs)
            self.conv2_point = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        if group:
            self.net = nn.Sequential(self.conv1, self.conv1_point, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                     self.conv2, self.conv2_point, self.chomp2, self.bn2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out+res
        out = self.relu(out)
        return out


class TemporalConvNet(nn.Module):
    """A Temporal Convolutional Network (TCN) for processing sequential data.

    This class implements a TCN architecture, stacking multiple TemporalBlock layers
    with increasing dilation factors to capture long-range dependencies in sequential
    data, such as EEG signals in brain-computer interface (BCI) tasks. It supports
    customizable input channels, output channels, kernel size, dropout, and normalization.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    num_inputs : int
        Number of input channels (e.g., number of EEG features).
    num_channels : list of int
        List of output channels for each TemporalBlock layer.
    kernel_size : int, optional
        Kernel size for the convolutional layers in TemporalBlock (default: 2).
    dropout : float, optional
        Dropout rate for regularization in TemporalBlock (default: 0.2).
    bias : bool, optional
        Whether to include bias terms in the convolutional layers (default: False).
    WeightNorm : bool, optional
        Whether to apply weight normalization to the convolutional layers (default: False).
    group : bool, optional
        Whether to apply group convolution in TemporalBlock (default: True).
    max_norm : float, optional
        Maximum norm constraint for the convolutional weights (default: 1.0).

    Attributes
    ----------
    network : torch.nn.Sequential
        A sequential container of TemporalBlock layers forming the TCN.

    Methods
    -------
    forward(x: torch.Tensor):
        Perform forward pass through the TCN.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, bias=False, WeightNorm=False, group=False, max_norm=1.):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout, bias=bias, WeightNorm=WeightNorm, group=group, max_norm=max_norm)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
### ==============================添加内容=============================== ###