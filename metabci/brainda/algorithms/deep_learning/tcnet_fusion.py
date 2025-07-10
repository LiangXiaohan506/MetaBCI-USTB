""" TCNet_Fusion model from Musallam et al 2021.
See details at https://doi.org/10.1016/j.bspc.2021.102826

    References
    ----------
    .. Musallam, Y.K., AlFassam, N.I., Muhammad, G., Amin, S.U., Alsulaiman,
        M., Abdul, W., Altaheri, H., Bencherif, M.A. and Algabri, M., 2021.
        Electroencephalography-based motor imagery classification
        using temporal convolutional network fusion.
        Biomedical Signal Processing and Control, 69, p.102826.
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
class TCNet_Fusion(nn.Module):
    """
    TCNet_Fusion is a deep learning model combining convolutional neural networks (CNNs)
    and temporal convolutional networks (TCNs) for EEG-based brain-computer interface (BCI)
    paradigms. The architecture integrates EEGNet-like convolutional layers with TCN blocks
    to capture both spatial and temporal features from EEG signals. It employs depthwise
    and separable convolutions to reduce parameters, along with batch normalization, ELU
    activation, and dropout for regularization.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

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
    step1 : torch.nn.Sequential
        Convolutional block including temporal, depthwise, and separable convolutions.
    step2 : torch.nn.Sequential
        Flattening layer for EEG features.
    step3 : torch.nn.Sequential
        Temporal Convolutional Network (TCraneNet) block.
    step4 : torch.nn.Sequential
        Flattening layer for combined EEG and TCN features.
    fc_layer : LinearWithConstraint
        Fully connected layer for classification.
    model : torch.nn.Sequential
        Complete model pipeline combining all steps.

    See Also
    --------
    _reset_parameters : Initialize model parameters with Glorot initialization.

    Examples
    --------
    >>> # Input size: [batch_size, n_channels, n_samples]
    >>> X = torch.randn(32, 22, 1000)
    >>> model = TCNet_Fusion(n_channels=22, n_samples=1000, n_classes=4)
    >>> model.fit(X[train_index], y[train_index]) # train the model using the fit method
    >>> output = model.forward_(input) # Forward propagation in the case of dual precision model parameters, used for training model parameters using a self defined model training method
    >>> model = model.module.to(dtype=torch.float32) # Forward propagation in the case of single precision model parameters, used for model inference
    >>> output = model(X) # Forward propagation in the case of single precision model parameters, used for model inference
    >>> print(output.shape)  # Expected: [32, 4]

    References
    ----------
    .. [1] Musallam Y K, AlFassam N I, Muhammad G, et al.
       Electroencephalography-based motor imagery classification using
       temporal convolutional network fusion[J].
       Biomedical Signal Processing and Control, 2021, 69: 102826.
    """
    def __init__(self, n_channels=22, n_samples=1000,
                 F1=12, D=2, kernLength=32, avgPoolKern=8, dropout_eeg=0.3,
                 tcn_filters=12, tcn_kernelSize=4, tcn_dropout=0.3, n_classes=4):
        super().__init__()
        F2 = F1*D

        self.step1 = nn.Sequential(
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

        self.step2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'flatten_eeg',
                        nn.Flatten()
                    )
                ]
            )
        )

        self.step3 = nn.Sequential(
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

        self.step4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'flatten_eeg_tcn',
                        nn.Flatten()
                    )
                ]
            )
        )

        self.fc_layer = LinearWithConstraint(
            in_features=n_samples//avgPoolKern//avgPoolKern*F2*2+n_samples//avgPoolKern//avgPoolKern*tcn_filters,
            out_features=n_classes,
            max_norm=.25
        )

        self.model = nn.Sequential(self.step1, self.step2, self.step3, self.step4, self.fc_layer)

        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        _glorot_weight_zero_bias(self)


    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)

        eeg_out = self.model[0](x)
        eeg_out = torch.squeeze(eeg_out, dim=2)
        eeg_out_fal = self.model[1](eeg_out)

        tcn_out = self.model[2](eeg_out)
        eeg_tcn_out = torch.cat((eeg_out, tcn_out), dim=1)
        eeg_tcn_out_fal = self.model[3](eeg_tcn_out)

        eeg_tcn_cat = torch.cat((eeg_out_fal, eeg_tcn_out_fal), dim=-1)
        out = self.model[4](eeg_tcn_cat)

        return out


    def cal_backbone(self, X: Tensor, **kwargs):
        X = X.unsqueeze(1)
        tmp = X

        eeg_out = self.model[0](tmp)
        eeg_out = torch.squeeze(eeg_out, dim=2)
        eeg_out_fal = self.model[1](eeg_out)

        tcn_out = self.model[2](eeg_out)
        eeg_tcn_out = torch.cat((eeg_out, tcn_out), dim=1)
        eeg_tcn_out_fal = self.model[3](eeg_tcn_out)

        eeg_tcn_cat = torch.cat((eeg_out_fal, eeg_tcn_out_fal), dim=-1)
        tmp = self.model[4](eeg_tcn_cat)

        return tmp