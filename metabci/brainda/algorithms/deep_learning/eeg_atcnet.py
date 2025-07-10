""" ATCNet model from Hamdi Altaheri et al 2022.
See details at https://10.1109/TII.2022.3197419

    References
    ----------
    .. Altaheri H, Muhammad G, Alsulaiman M.
    Physics-informed attention temporal convolutional network for EEG-based motor imagery classification[J].
    IEEE transactions on industrial informatics, 2022, 19(2): 2249-2258.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor
from .base import (
    Conv2dWithConstraint,
    LinearWithConstraint,
    TemporalConvNet,
    SkorchNet
)

@SkorchNet
class ATCNet(nn.Module):
    """
    ATCNet is a deep learning model combining convolutional neural networks (CNNs),
    temporal convolutional networks (TCNs), and attention mechanisms for EEG-based
    brain-computer interface (BCI) paradigms. The architecture integrates EEGNet-like
    convolutional layers, multi-head attention, and TCN blocks with a sliding window
    approach to capture spatial and temporal features from EEG signals. It employs
    depthwise and separable convolutions, batch normalization, ELU activation, and
    dropout for regularization.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-06-11

    update log:
        2025-06-11 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    n_classes : int, optional
        Number of classes for classification (default: 4).
    in_chans : int, optional
        Number of EEG channels (default: 22).
    in_samples : int, optional
        Number of time points per sample (default: 1125).
    n_windows : int, optional
        Number of sliding windows (default: 5).
    eegn_F1 : int, optional
        Number of filters in the first convolutional layer (default: 16).
    eegn_D : int, optional
        Depth multiplier for depthwise convolution (default: 2).
    eegn_kernelSize : int, optional
        Kernel length for temporal convolution (default: 64).
    eegn_poolSize : int, optional
        Pooling size for average pooling (default: 7).
    eegn_dropout : float, optional
        Dropout rate for EEG convolutional layers (default: 0.3).
    tcn_depth : int, optional
        Number of TCN layers (default: 2).
    tcn_kernelSize : int, optional
        Kernel size for TCN layers (default: 4).
    tcn_filters : int, optional
        Number of filters in TCN layers (default: 32).
    tcn_dropout : float, optional
        Dropout rate for TCN layers (default: 0.3).
    fuse : str, optional
        Fusion method for sliding window outputs ('average' or 'concat', default: 'average').

    Attributes
    ----------
    step1 : torch.nn.Sequential
        Convolutional block including temporal, depthwise, and separable convolutions.
    step2 : torch.nn.ModuleList
        List of attention and TCN blocks for sliding windows.
    step3 : torch.nn.ModuleList
        List of dense layers for sliding window outputs (if fuse='average').
    step4 : torch.nn.Sequential
        Flattening layer for fused features.
    fc_layer : LinearWithConstraint
        Fully connected layer for classification (if fuse='concat').
    model : torch.nn.Module
        Complete model pipeline combining all steps.

    See Also
    --------
    _reset_parameters : Initialize model parameters with Glorot initialization.

    Examples
    --------
    >>> # Input size: [batch_size, n_channels, n_samples]
    >>> import torch
    >>> X = torch.randn(32, 22, 1125)
    >>> model = ATCNet(n_classes=4, in_chans=22, in_samples=1125)
    >>> model.fit(X[train_index], y[train_index]) # train the model using the fit method
    >>> output = model.forward_(input) # Forward propagation in the case of dual precision model parameters, used for training model parameters using a self defined model training method
    >>> model = model.module.to(dtype=torch.float32) # Forward propagation in the case of single precision model parameters, used for model inference
    >>> output = model(X) # Forward propagation in the case of single precision model parameters, used for model inference
    >>> print(output.shape)  # Expected: [32, 4]

    References
    ----------
    .. [1] H. Altaheri, G. Muhammad, and M. Alsulaiman. "Physics-informed
           attention temporal convolutional network for EEG-based motor imagery
           classification." IEEE Transactions on Industrial Informatics,
           vol. 19, no. 2, pp. 2249-2258, (2023)
           https://doi.org/10.1109/TII.2022.3197419
    """
    def __init__(self, n_classes=4, n_channels=22, n_samples=1125, n_windows=5,
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 fuse='average'):
        super().__init__()
        F2 = eegn_F1 * eegn_D
        self.n_windows = n_windows
        self.fuse = fuse
        self.in_samples = n_samples

        # Step 1: Convolutional block
        self.step1 = nn.Sequential(
            OrderedDict([
                ('time_conv', Conv2dWithConstraint(
                    in_channels=1, out_channels=eegn_F1, kernel_size=(1, eegn_kernelSize),
                    padding='same', bias=False, max_norm=0.6
                )),
                ('time_bn', nn.BatchNorm2d(eegn_F1)),
                ('depthwise_conv', nn.Conv2d(
                    in_channels=eegn_F1, out_channels=F2, kernel_size=(n_channels, 1),
                    groups=eegn_F1, bias=False
                )),
                ('depth_bn', nn.BatchNorm2d(F2)),
                ('depth_elu', nn.ELU()),
                ('depth_pool', nn.AvgPool2d(kernel_size=(1, 8))),
                ('depth_dropout', nn.Dropout(eegn_dropout)),
                ('separable_conv', Conv2dWithConstraint(
                    in_channels=F2, out_channels=F2, kernel_size=(1, 16),
                    padding='same', bias=False, max_norm=0.6
                )),
                ('separable_bn', nn.BatchNorm2d(F2)),
                ('separable_elu', nn.ELU()),
                ('separable_pool', nn.AvgPool2d(kernel_size=(1, eegn_poolSize))),
                ('separable_dropout', nn.Dropout(eegn_dropout))
            ])
        )

        # Step 2: Attention and TCN blocks for sliding windows
        self.step2 = nn.ModuleList([
            nn.Sequential(
                OrderedDict([
                    ('attention', nn.MultiheadAttention(
                        embed_dim=F2, num_heads=2, dropout=0.5
                    )),
                    ('attention_norm', nn.LayerNorm(F2, eps=1e-6)),
                    ('attention_dropout', nn.Dropout(0.3)),
                    ('tcn', TemporalConvNet(
                        num_inputs=F2, num_channels=[tcn_filters] * tcn_depth,
                        kernel_size=tcn_kernelSize, dropout=tcn_dropout
                    ))
                ])
            ) for _ in range(n_windows)
        ])

        # Step 3: Dense layers for sliding window outputs (if fuse='average')
        self.step3 = nn.ModuleList([
            LinearWithConstraint(
                in_features=tcn_filters, out_features=n_classes, max_norm=0.25
            ) for _ in range(n_windows)
        ]) if fuse == 'average' else None

        # Step 4: Flattening and fusion
        self.step4 = nn.Sequential(
            OrderedDict([
                ('flatten', nn.Flatten())
            ])
        )

        # Final classification layer (if fuse='concat')
        self.fc_layer = LinearWithConstraint(
            in_features=tcn_filters * n_windows, out_features=n_classes, max_norm=0.25
        ) if fuse == 'concat' else None


    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)  # [batch_size, 1, in_chans, in_samples]

        # Step 1: Convolutional block
        x = self.step1(x)  # [batch_size, F2, 1, time]
        x = torch.squeeze(x, dim=2)  # [batch_size, F2, time]
        # x = x[:, :, -1:]  # [batch_size, F2, 1]

        # Step 2 & 3: Sliding window with attention and TCN
        sw_outputs = []
        for i in range(self.n_windows):
            st = i
            end = x.shape[2] - self.n_windows + i + 1
            window = x[:, :, st:end]  # [batch_size, F2, window_size]
            window = window.permute(2, 0, 1)  # [window_size, batch_size, F2]
            attn_output, _ = self.step2[i][0](window, window, window)
            attn_output = self.step2[i][1](attn_output)
            attn_output = self.step2[i][2](attn_output)
            tcn_output = self.step2[i][3](attn_output.permute(1, 2, 0))  # [batch_size, tcn_filters, time]
            tcn_output = tcn_output[:, :, -1]  # [batch_size, tcn_filters]
            if self.fuse == 'average':
                sw_outputs.append(self.step3[i](tcn_output))
            else:
                sw_outputs.append(tcn_output)

        # Fusion
        if self.fuse == 'average':
            if len(sw_outputs) > 1:
                sw_concat = torch.mean(torch.stack(sw_outputs), dim=0)
            else:
                sw_concat = sw_outputs[0]
        else:
            sw_concat = torch.cat(sw_outputs, dim=1)
            sw_concat = self.step4(sw_concat)
            sw_concat = self.fc_layer(sw_concat)

        return sw_concat

    def cal_backbone(self, x: Tensor, **kwargs) -> Tensor:
        x = x.unsqueeze(1)  # [batch_size, 1, in_chans, in_samples]

        # Step 1: Convolutional block
        x = self.step1(x)  # [batch_size, F2, 1, time]
        x = torch.squeeze(x, dim=2)  # [batch_size, F2, time]
        x = x[:, :, -1:]  # [batch_size, F2, 1]

        # Step 2 & 3: Sliding window with attention and TCN
        sw_outputs = []
        for i in range(self.n_windows):
            st = i
            end = x.shape[2] - self.n_windows + i + 1
            window = x[:, :, st:end]  # [batch_size, F2, window_size]
            window = window.permute(2, 0, 1)  # [window_size, batch_size, F2]
            attn_output, _ = self.step2[i][0](window, window, window)
            attn_output = self.step2[i][1](attn_output)
            attn_output = self.step2[i][2](attn_output)
            tcn_output = self.step2[i][3](attn_output.permute(1, 2, 0))  # [batch_size, tcn_filters, time]
            tcn_output = tcn_output[:, :, -1]  # [batch_size, tcn_filters]
            if self.fuse == 'average':
                sw_outputs.append(self.step3[i](tcn_output))
            else:
                sw_outputs.append(tcn_output)

        # Fusion
        if self.fuse == 'average':
            if len(sw_outputs) > 1:
                sw_concat = torch.mean(torch.stack(sw_outputs), dim=0)
            else:
                sw_concat = sw_outputs[0]
        else:
            sw_concat = torch.cat(sw_outputs, dim=1)
            sw_concat = self.step4(sw_concat)
            sw_concat = self.fc_layer(sw_concat)

        return sw_concat
