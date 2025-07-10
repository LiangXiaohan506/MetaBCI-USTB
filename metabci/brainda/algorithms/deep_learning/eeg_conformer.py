""" EEG_Conformer model from Song Y et al 2022.
See details at https://10.1109/SMC42975.2020.9283028

    References
    ----------
    .. Song Y, Zheng Q, Liu B, et al.
    EEG conformer: Convolutional transformer for EEG decoding and visualization[J].
    IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2022, 31: 710-719.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, reduce
from .base import SkorchNet, _glorot_weight_zero_bias


@SkorchNet
class EEG_Conformer(nn.Module):
    """
    Convolutional Transformer for EEG decoding

    Combines CNN's local feature extraction with Transformer's global context modeling,
    achieving state-of-the-art performance on major EEG datasets.

    Parameters
    ----------
    n_classes: int
        Number of output classes.
    Chans: int
        Number of EEG channels.
    Samples: int
        Number of time samples.
    emb_size: int
        Transformer embedding dimension.
    depth: int
        Number of Transformer encoder layers.

    Attributes
    ----------
    patch_embed: torch.nn.Sequential
        Convolutional block for converting EEG signals into embeddings.
    transformer: torch.nn.Sequential
        Transformer encoder for capturing global context.
    cls_head: torch.nn.Sequential
        Classification head for mapping Transformer outputs to class scores.
    model: torch.nn.Sequential
        Complete model architecture.

    Examples
    ----------
    >>> # X size: [batch size, 1, number of channels, number of sample points]
    >>> num_classes = 2
    >>> estimator = Conformer(n_classes=num_classes, Chans=X.shape[2], Samples=X.shape[3])
    >>> estimator.fit(X[train_index], y[train_index])

    References
    ----------
    .. [1] Song et al. "EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization"
           IEEE Transactions on Neural Systems and Rehabilitation Engineering (2023).
    """
    def __init__(self, n_classes=4, n_channels=22, n_samples=1000, emb_size=40, depth=6):
        super().__init__()
        self.Chans = n_channels
        self.Samples = n_samples
        self.emb_size = emb_size
        self.depth = depth

        self.model = nn.Sequential(OrderedDict([
            ('patch_embed', self._make_patch_embed(emb_size)),
            ('transformer', self._make_transformer(emb_size, depth)),
            ('cls_head', self._make_classifier(emb_size, n_classes))
        ]))


    def _make_patch_embed(self, emb_size):
        """
        Create the patch embedding layer.
        """
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 40, (1, 25), (1, 1))),
            ('conv2', nn.Conv2d(40, 40, (self.Chans, 1), (1, 1), groups=40)),
            ('bn', nn.BatchNorm2d(40)),
            ('elu', nn.ELU()),
            ('avg_pool', nn.AvgPool2d((1, 75), (1, 15))),
            ('dropout', nn.Dropout(0.5)),
            ('conv3', nn.Conv2d(40, emb_size, (1, 1), (1, 1))),
            ('rearrange', Rearrange('b e h w -> b (h w) e'))
        ]))

    def _make_transformer(self, emb_size, depth):
        """
        Create the Transformer encoder.
        """
        encoder_blocks = []
        for _ in range(depth):
            encoder_blocks.append(('norm1', nn.LayerNorm(emb_size)))
            encoder_blocks.append(('attn', MultiHeadAttention(emb_size, num_heads=10, dropout=0.5)))
            encoder_blocks.append(('norm2', nn.LayerNorm(emb_size)))
            encoder_blocks.append(('ffn', FeedForwardBlock(emb_size, expansion=4, drop_p=0.5)))
        return nn.Sequential(OrderedDict(encoder_blocks))

    def _make_classifier(self, emb_size, n_classes):
        """
        Create the classification head.
        """
        return nn.Sequential(OrderedDict([
            ('reduce', Reduce('b n e -> b e', reduction='mean')),
            ('norm', nn.LayerNorm(emb_size)),
            ('linear', nn.Linear(emb_size, n_classes))
        ]))


    def forward(self, x: Tensor, visual=False) -> Tensor:
        """
        Forward pass through the network.
        """
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)
        if visual:
            embed_feature = self.model[0](x)
            transformer_feature = self.model[1](embed_feature)
            out = self.model[2:](transformer_feature)
            return out, torch.flatten(embed_feature, start_dim=1), torch.flatten(transformer_feature, start_dim=1)
        else:
            out = self.model(x)
            return out

    def cal_backbone(self, x: Tensor, **kwargs):
        """
        cal_backbone pass through the network.
        """
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)
        tmp = self.model(x)
        return tmp


class MultiHeadAttention(nn.Module):
    """
    Multi-head self attention with projection.
    """
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att_drop = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att_drop, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        # import os
        # import pickle
        # features_path_2 = "checkpoints\\EEG_Conformer\\visualization\\attention_feature\\"
        # if not os.path.exists(features_path_2):
        #     os.makedirs(features_path_2)
        # features_path_2 = features_path_2 + "attention_feature.pkl"
        # with open(features_path_2, 'wb') as f:
        #     pickle.dump(att[0].to('cpu').numpy(), f)

        return out


class FeedForwardBlock(nn.Sequential):
    """
    Position-wise feed forward network.
    """
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )


class Rearrange(nn.Module):
    """
    Custom module to wrap einops.rearrange for use in nn.Sequential.
    """
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(x, self.pattern)


class Reduce(nn.Module):
    """
    Custom module to perform reduction operation (e.g., mean) along specified dimensions.
    """
    def __init__(self, pattern, reduction='mean'):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction

    def forward(self, x: Tensor) -> Tensor:
        return reduce(x, self.pattern, reduction=self.reduction)