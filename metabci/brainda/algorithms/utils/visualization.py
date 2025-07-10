"""
Copyright (C) 2023 Qufu Normal University, Guangjin Liang
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Author:  Guangjin Liang <3330635482@qq.com>
Update log: 2025-05-21 by Guangjin Liang <3330635482@qq.com>
Modified from https://github.com/LiangXiaohan506/EISATC-Fusion/blob/main/visualization.py
"""

import os
import mne
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay



def confusion_matrix_disply(confusion, labels, output_dir, dpi=300):
    """Visualize and save a confusion matrix for classification performance evaluation.

    This function creates a formatted confusion matrix visualization using a provided
    confusion matrix and class labels, customized with a specific font, grid, and color scheme.
    The matrix is displayed with percentage values and saved as a high-resolution PNG image.
    It is designed for evaluating brain-computer interface (BCI) classification models, such as
    those used in EEG signal classification.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    confusion : np.ndarray
        Confusion matrix with shape [n_classes, n_classes], where element (i, j) represents
        the number of samples with true label i predicted as j.
    labels : list of str
        List of class labels for the confusion matrix axes.
    output_dir : str
        Directory path or file path prefix to save the confusion matrix image.
        The file will be saved as 'Confusion Matrix.png' appended to this path.
    dpi : int, optional
        Resolution of the saved image in dots per inch (default: 300).

    Returns
    -------
    None
        Displays the confusion matrix plot and saves it as a PNG file at the specified path.
        Prints a confirmation message with the save location.
    """
    plt.rc('font',family='Times New Roman') # 设置画布的全局字体
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion)

    # Plot the confusion matrix and get the Axes object
    ax = disp.plot(include_values=True, cmap=plt.cm.Reds, values_format='.2%', colorbar=True, ax=None)

    # Add minor ticks at each cell boundary
    x_ticks = np.arange(-.5, len(labels), 1)
    y_ticks = np.arange(-.5, len(labels), 1)
    ax.ax_.set_xticks(x_ticks, minor=True)
    ax.ax_.set_yticks(y_ticks, minor=True)

    ax.ax_.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    # Add grid for minor ticks
    ax.ax_.grid(True, which='minor', color='white', linestyle='-', linewidth=1.5)

    ax.ax_.set_yticklabels(labels, va='center')
    ax.ax_.set_xticklabels(labels, va='center')
    ax.ax_.xaxis.set_tick_params(pad=10)

    plt.yticks(rotation=90)

    output_dir = output_dir + 'Confusion Matrix'+'.png'
    plt.savefig(output_dir, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: '+output_dir)
    plt.show()


def plot_tsne_feature(features_path, true_labels, output_dir, file_name, dpi=300, random_state=44):
    """Visualize high-dimensional features using t-SNE and save as a scatter plot.

    This function loads high-dimensional feature data from a pickle file, applies t-SNE
    to reduce the dimensionality to 2D, and generates a scatter plot where points are
    colored by their true labels. The plot is saved as a high-resolution PNG image in
    the specified output directory. It is designed for analyzing feature distributions
    in brain-computer interface (BCI) tasks, such as EEG signal classification.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    features_path : str
        Path to the pickle file containing high-dimensional feature data.
    true_labels : array-like
        True labels for the feature data, shape [n_samples], used to color the scatter plot.
    output_dir : str
        Directory path or prefix for saving the t-SNE scatter plot.
        The file will be saved as 'featre_tSNE.png' in a subdirectory 't-sne figure/'.
    file_name : str
        Name of the file to save the t-SNE scatter plot.
    dpi : int, optional
        Resolution of the saved image in dots per inch (default: 300).
    random_state : int, optional
        Random seed for t-SNE reproducibility (default: 44).

    Returns
    -------
    None
        Generates and displays a t-SNE scatter plot, saves it as a PNG file, and prints
        a confirmation message with the save location.
    """
    # Load feature data from pickle file
    print('Loading features ……')
    with open(features_path, "rb") as features:
        Fatures = pickle.load(features)
    print('Fatures loading complete!')

    # Initialize t-SNE and plot
    print('Image is being generated……')
    colors = [5, 3, 1, 7] # Predefined color indices for plt.cm.Paired

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=5, n_iter=3000, n_iter_without_progress=300, random_state=random_state)
    plt.figure(figsize=(5, 5)) # Create figure for scatter plot
    X_tsne = tsne.fit_transform(Fatures) # Transform features to 2D
    X_tsne = MinMaxScaler().fit_transform(X_tsne) # Normalize t-SNE output to [0, 1] range

    # Plot scatter points for each class
    for category in np.unique(true_labels):
            plt.scatter(
            *X_tsne[true_labels == category].T,
            marker=".", # f"${digit}$",
            color=plt.cm.Paired(colors[int(category)]),
            # label=labels[int(category)],
            alpha=0.8,
            s=100)
    # Configure axes to hide ticks and labels
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.legend() # Add legend
    # Construct output directory for t-SNE figure
    output_dir_file = output_dir + 't-sne figure/'
    if not os.path.exists(output_dir_file):
        os.makedirs(output_dir_file)
    output_dir_name = output_dir_file + file_name # Construct output file path
    plt.savefig(output_dir_name, dpi=dpi) # Save the plot as a high-resolution image
    print(f'The picture is saved successfully!\nSave address: '+output_dir_name)
    plt.show()


def convWeight_to_waveform(model_savePath, model_convLayerName, output_dir, ylim=1, scalingCol=3, scalingRow=2, dpi=300):
    """Visualize convolutional layer weights as time-series waveforms and save as an image.

    This function loads the weights of a specified convolutional layer from a saved PyTorch
    model, extracts the temporal convolutional kernel weights, and plots them as time-series
    waveforms in a grid layout. The resulting plot is saved as a high-resolution PNG image.
    It is designed for analyzing the learned filters of convolutional neural networks in
    brain-computer interface (BCI) tasks, such as EEG signal classification with models like
    TCNet_Fusion or EEGNet.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    model_savePath : str
        Path to the saved PyTorch model file (.pth) containing the model state dictionary.
    model_convLayerName : str
        Name of the convolutional layer (e.g., 'conv1.weight') whose weights are to be visualized.
    output_dir : str
        Directory path or prefix for saving the waveform plot.
        The file will be saved as 'Weight_waveform.png' in this directory.
    ylim : float, optional
        Absolute limit for the y-axis of each subplot, setting the range to [-ylim, ylim]
        (default: 1).
    scalingCol : float, optional
        Scaling factor for the width of each subplot (default: 3).
    scalingRow : float, optional
        Scaling factor for the height of each subplot (default: 2).
    dpi : int, optional
        Resolution of the saved image in dots per inch (default: 300).

    Returns
    -------
    None
        Generates and displays a grid of waveform plots for the convolutional weights,
        saves the plot as a PNG file, and prints a confirmation message with the save location.
    """
    # Load model parameters
    print('Loading model parameters……')
    state_dict = torch.load(model_savePath)
    print('The model parameters are loaded!')
    print('Image is being generated……')
    # Extract temporal convolutional weights
    temporalConv_wieght = state_dict[model_convLayerName].cpu().numpy()
    # Create figure with subplots based on weight dimensions
    fig, axs = plt.subplots(nrows=temporalConv_wieght.shape[0],
                            ncols=temporalConv_wieght.shape[1],
                            figsize=(temporalConv_wieght.shape[1]*scalingCol, temporalConv_wieght.shape[0]*scalingRow))
    # Plot waveforms for each filter
    if len(axs.shape) == 1:
        axs = axs[:, np.newaxis]
    for i in range(temporalConv_wieght.shape[0]):
        for j in range(temporalConv_wieght.shape[1]):
            axs[i, j].plot(temporalConv_wieght[i,j,0,:])
            # axs[i//8, i%8].set_xlim(0,0.125)
            axs[i, j].set_ylim(-ylim, ylim)
    # Create output directory if it does not exist
    output_dir_file = output_dir + 'waveform figure/'
    if not os.path.exists(output_dir_file):
        os.makedirs(output_dir_file)
    # Construct output file path
    output_dir_name = output_dir_file + "Weight_waveform.png"
    # Save the plot as a high-resolution image
    plt.savefig(output_dir_name, dpi=dpi)
    # Print confirmation message with save path
    print(f'The picture is saved successfully!\nSave address: '+output_dir_name)
    # Display the plot
    plt.show()


def convWeight_to_topography(model_savePath, model_convLayerName, output_dir, scalingCol=3, scalingRow=2, dpi=300, channelsName=None):
    """Visualize depth convolutional layer weights as EEG topographic maps and save as an image.

    This function loads the weights of a specified depth convolutional layer from a saved PyTorch
    model, extracts the weights, and visualizes them as EEG topographic maps using the standard
    10-20 electrode montage. The maps are arranged in a grid layout and saved as a high-resolution
    PNG image. It is designed for analyzing the spatial patterns of convolutional filters in
    brain-computer interface (BCI) tasks, such as EEG signal classification with models like
    TCNet_Fusion or EEGNet.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    model_savePath : str
        Path to the saved PyTorch model file (.pth) containing the model state dictionary.
    model_convLayerName : str
        Name of the depth convolutional layer (e.g., 'conv_depth.weight') whose weights are to be visualized.
    output_dir : str
        Directory path or prefix for saving the topographic plot.
        The file will be saved as 'Weight_topo.png' in this directory.
    scalingCol : float, optional
        Scaling factor for the width of each subplot (default: 3).
    scalingRow : float, optional
        Scaling factor for the height of each subplot (default: 2).
    dpi : int, optional
        Resolution of the saved image in dots per inch (default: 300).
    channelsName : list, optional
        List of channel names for the EEG topographic plot (default: None).

    Returns
    -------
    None
        Generates and displays a grid of EEG topographic plots for the convolutional weights,
        saves the plot as a PNG file with a colorbar, and prints a confirmation message with
        the save location.
    """
    print('Loading model parameters……')
    # 加载模型参数
    state_dict = torch.load(model_savePath)
    print('The model parameters are loaded!')
    print('Image is being generated……')

    # 读取深度卷积层的卷积核权重
    depthConv_wieght = state_dict[model_convLayerName].cpu().numpy()

    biosemi_montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=channelsName, sfreq=1., ch_types='eeg')
    info.set_montage(biosemi_montage)

    # 设置画布行列数和画布的尺寸
    fig, axs = plt.subplots(nrows=depthConv_wieght.shape[0],
                           ncols=depthConv_wieght.shape[1],
                           figsize=(depthConv_wieght.shape[1]*scalingCol, depthConv_wieght.shape[0]*scalingRow))
    # 在画布上绘制32个地形图
    if len(axs.shape) == 1:
        axs = axs[:, np.newaxis]
    for i in range(depthConv_wieght.shape[0]):
        for j in range(depthConv_wieght.shape[1]):
            im, cn = mne.viz.plot_topomap(depthConv_wieght[i, j, :, 0], info, show=False, axes=axs[i, j], extrapolate='local')
            # ax[i%4,i//4].set(title="Model coefficients\nbetween delays  and ")
    # 添加颜色棒
    fig.colorbar(im, ax=axs.ravel().tolist())
    # Create output directory if it does not exist
    output_dir_file = output_dir + 'topography figure/'
    if not os.path.exists(output_dir_file):
        os.makedirs(output_dir_file)
    output_dir_name = output_dir_file + "Weight_topography.png"
    plt.savefig(output_dir_name, dpi=dpi)
    print(f'The picture is saved successfully!\nSave address: '+output_dir_name)
    plt.show()


### 模型可解释性——可视化处理——一键生成cnnCosMSA的注意力信息效果图
def attentionWeight_Visualization(attention_savePath, output_dir, scalingCol=3, scalingRow=2, dpi=300):
    """Visualize attention weights as heatmaps and save as an image.

    This function loads attention weights from a pickle file, typically representing the
    attention scores of a transformer-based model, and visualizes each attention head as a
    heatmap in a single-row grid layout. The heatmaps are saved as a high-resolution PNG image
    in a specified output directory. It is designed for analyzing attention mechanisms in
    brain-computer interface (BCI) tasks, such as EEG signal classification with transformer
    models integrated in architectures like TCNet_Fusion.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    attention_savePath : str
        Path to the pickle file containing attention weights, typically a numpy array with
        shape [n_heads, height, width].
    output_dir : str
        Directory path or prefix for saving the attention heatmap plot.
        The file will be saved as 'Attention_weight.png' in a subdirectory 'attention figure/'.
    scalingCol : float, optional
        Scaling factor for the width of each subplot (default: 3).
    scalingRow : float, optional
        Scaling factor for the height of the subplot row (default: 2).
    dpi : int, optional
        Resolution of the saved image in dots per inch (default: 300).

    Returns
    -------
    None
        Generates and displays a row of heatmap plots for the attention weights,
        saves the plot as a PNG file, and prints a confirmation message with the save location.
    """
    # Set global font for the plot
    plt.rc('font',family='Times New Roman')
    # Load attention weights from pickle file
    print('Loading model parameters……')
    with open(attention_savePath, "rb") as attentionWeightFile:
        dattentionWeight = pickle.load(attentionWeightFile)
    print('Data loading complete!')

    # Create figure with subplots for each attention head
    fig, axs = plt.subplots(nrows=1,
                            ncols=dattentionWeight.shape[0],
                            figsize=(dattentionWeight.shape[0]*scalingCol, 1*scalingRow))

    # Plot heatmap for each attention head
    if len(axs.shape) == 1:
        axs = axs[np.newaxis, :]
    for head in range(dattentionWeight.shape[0]):
        # Display attention weights as a heatmap
        axs[0, head].imshow(dattentionWeight[head], cmap=plt.cm.Reds) # , vmax=1., vmin=0.
        # Set title with head number and mean attention weight
        axs[0, head].set_title("head {:}".format(head+1), fontdict={'size':16, 'weight':'bold'}) #

    # Construct output directory and file path
    output_dir_file = output_dir + 'attention figure/'
    if not os.path.exists(output_dir_file):
        os.makedirs(output_dir_file)
    output_dir_name = output_dir_file + 'Attention_weight.png'
    # Save the plot as a high-resolution image
    plt.savefig(output_dir_name, dpi=dpi)
    # Print confirmation message with save path
    print(f'The picture is saved successfully!\nSave address: ' + output_dir_name)
    # Display the plot
    plt.show()



