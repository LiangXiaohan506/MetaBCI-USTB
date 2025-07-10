""" EEG-TCNet model from Ingolfsson et al 2020.
See details at https://10.1109/SMC42975.2020.9283028

    References
    ----------
    .. Ingolfsson T M, Hersche M, Wang X, et al. EEG-TCNet:
    An accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces[C]
    //2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, 2020: 2958-2965.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor
from .base import (
    Conv2dWithConstraint,
    LinearWithConstraint,
    TemporalConvNet,
    _glorot_weight_zero_bias,
    SkorchNet
)


@SkorchNet
class EEG_TCNet(nn.Module):
    """
    EEG_TCNet is a deep learning model combining convolutional neural networks (CNNs)
    and temporal convolutional networks (TCNs) for EEG-based brain-computer interface (BCI)
    paradigms. The architecture integrates EEGNet-like convolutional layers with TCN blocks
    to capture both spatial and temporal features from EEG signals. It employs depthwise
    and separable convolutions to reduce parameters, along with batch normalization, ELU
    activation, and dropout for regularization.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-06-11

    update log:
        2025-06-11 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    n_channels : int, optional
        Number of EEG channels (default: 22).
    n_samples : int, optional
        Number of time points per sample (default: 1000).
    F1 : int, optional
        Number of filters in the first convolutional layer (default: 24).
    D : int, optional
        Depth multiplier for depthwise convolution (default: 2).
    kernLength : int, optional
        Kernel length for temporal convolution (default: 32).
    dropout_eeg : float, optional
        Dropout rate for EEG convolutional layers (default: 0.3).
    tcn_filters : int, optional
        Number of filters in TCN layers (default: 12).
    tcn_kernelSize : int, optional
        Kernel size for TCN layers (default: 4).
    tcn_dropout : float, optional
        Dropout rate for TCN layers (default: 0.3).
    n_classes : int, optional
        Number of classes for classification (default: 4).

    Attributes
    ----------
    eegNet : torch.nn.Sequential
        Convolutional block including temporal, depthwise, and separable convolutions.
    tcnBlock : torch.nn.Sequential
        Temporal Convolutional Network (TCN) block for temporal feature extraction.
    flatten : torch.nn.Sequential
        Flattening layer for EEG and TCN features.
    classifier : LinearWithConstraint
        Fully connected layer for classification with max-norm constraint.
    model : torch.nn.Sequential
        Complete model pipeline combining all steps.

    See Also
    --------
    _reset_parameters : Initialize model parameters with Glorot initialization.

    Examples
    --------
    >>> # Input size: [batch_size, 1, n_channels, n_samples]
    >>> X = torch.randn(32, 1, 22, 1000)
    >>> model = EEG_TCNet(n_channels=22, n_samples=1000, n_classes=4)
    >>> model.fit(X[train_index], y[train_index]) # train the model using the fit method
    >>> output = model.forward_(input) # Forward propagation in the case of dual precision model parameters, used for training model parameters using a self defined model training method
    >>> model = model.module.to(dtype=torch.float32) # Forward propagation in the case of single precision model parameters, used for model inference
    >>> output = model(X) # Forward propagation in the case of single precision model parameters, used for model inference
    >>> print(output.shape)  # Expected: [32, 4]

    References
    ----------
    .. [1] Ingolfsson T M, Hersche M, Wang X, et al. EEG-TCNet:
           An accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces[C]
           //2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, 2020: 2958-2965.
    """
    def __init__(self, n_channels=22, n_samples=1000,
                 F1=12, D=2, kernLength=32, avgPoolKern=8, dropout_eeg=0.3,
                 tcn_filters=12, tcn_kernelSize=4, tcn_dropout=0.3, n_classes=4):
        super().__init__()
        F2 = F1 * D

        self.eegNet = nn.Sequential(
            OrderedDict(
                [
                    (
                        'time_conv',
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=F1,
                            kernel_size=(1, kernLength),
                            stride=1,
                            padding='same',
                            bias=False
                        )
                    ),
                    ('time_bn', nn.BatchNorm2d(num_features=F1)),

                    (
                        'depthwise_conv',
                        Conv2dWithConstraint(
                            in_channels=F1,
                            out_channels=F1 * D,
                            kernel_size=(n_channels, 1),
                            groups=F1,
                            bias=False,
                            max_norm=1.
                        )
                    ),
                    ('dept_bn', nn.BatchNorm2d(num_features=F1 * D)),
                    ('dept_elu', nn.ELU()),
                    ('dept_pool', nn.AvgPool2d(kernel_size=(1, avgPoolKern), stride=(1, avgPoolKern))),
                    ('dept_dropout', nn.Dropout(p=dropout_eeg)),

                    (
                        'seqarableConv_1',
                        nn.Conv2d(
                            in_channels=F2,
                            out_channels=F2,
                            kernel_size=(1, 16),
                            stride=1,
                            padding='same',
                            groups=F2,
                            bias=False
                        )
                    ),
                    (
                        'seqarableConv_2',
                        nn.Conv2d(
                            in_channels=F2,
                            out_channels=F2,
                            kernel_size=(1, 1),
                            stride=1,
                            bias=False
                        )
                    ),
                    ('seqarable_bn', nn.BatchNorm2d(num_features=F2)),
                    ('seqarable_elu', nn.ELU()),
                    ('seqarable_pool', nn.AvgPool2d(kernel_size=(1, avgPoolKern), stride=(1, avgPoolKern))),
                    ('seqarable_dropout', nn.Dropout(p=dropout_eeg))
                ]
            )
        )

        self.tcnBlock = nn.Sequential(
            OrderedDict(
                [
                    (
                        'tcn_block',
                        TemporalConvNet(
                            num_inputs=F2,
                            num_channels=[tcn_filters, tcn_filters],
                            kernel_size=tcn_kernelSize,
                            dropout=tcn_dropout,
                            group=True,
                            max_norm=1.
                        )
                    )
                ]
            )
        )

        self.flatten = nn.Sequential(
            OrderedDict(
                [
                    (
                        'flatten_tcn',
                        nn.Flatten()
                    )
                ]
            )
        )

        # Classifier
        self.classifier = LinearWithConstraint(
            in_features=n_samples//avgPoolKern//avgPoolKern*tcn_filters,
            out_features=n_classes,
            max_norm=.25
        )

        # Complete model
        self.model = nn.Sequential(self.eegNet, self.tcnBlock, self.flatten, self.classifier)

        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        _glorot_weight_zero_bias(self)

    def forward(self, x: Tensor):
        """
        Forward pass through the network.

        Parameters
        ----------
        X: Tensor
            Input tensor of shape [batch_size, n_channels, n_samples]

        Returns
        -------
        Tensor
            Output tensor of shape [batch_size, n_classes]
        """
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)

        eeg_out = self.model[0](x)
        eeg_out = torch.squeeze(eeg_out, dim=2)
        tcn_out = self.model[1](eeg_out)
        tcn_out_fal = self.model[2](tcn_out)
        out = self.model[3](tcn_out_fal)
        return out

    def cal_backbone(self, x: Tensor, **kwargs):
        """
        Calculate the output of the backbone network (before classification layer).

        Parameters
        ----------
        X: Tensor
            Input tensor of shape [batch_size, n_channels, n_samples]

        Returns
        -------
        Tensor
            Output tensor of shape [batch_size, n_features]
        """
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)

        eeg_out = self.model[0](x)
        eeg_out = torch.squeeze(eeg_out, dim=2)
        tcn_out = self.model[1](eeg_out)
        tcn_out_fal = self.model[2](tcn_out)
        tmp = self.model[3](tcn_out_fal)
        return tmp


