# -*- coding: utf-8 -*-
# License: MIT License
"""
MI offline analysis.

"""
import os
import torch
import torch.nn as nn
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from metabci.brainda.algorithms.decomposition.csp import FBCSP
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.feature_analysis.time_freq_analysis import TimeFrequencyAnalysis
from metabci.brainda.algorithms.feature_analysis.freq_analysis import FrequencyAnalysis
from metabci.brainda.algorithms.feature_analysis.entropy_analysis import EntropyAnalysis
from metabci.brainda.algorithms.utils.model_selection import EnhancedLeaveOneGroupOut

from metabci.brainda.algorithms.deep_learning import EEGNet
from metabci.brainda.algorithms.deep_learning import TCNet_Fusion
from metabci.brainda.algorithms.deep_learning import EEG_Conformer
from metabci.brainda.algorithms.deep_learning import EEG_TCNet
from metabci.brainda.algorithms.deep_learning import ATCNet

from metabci.brainda.algorithms.utils.model_selection import model_training_two_stage
from metabci.brainda.algorithms.utils.model_selection import model_training_two_stage_up
from metabci.brainda.algorithms.utils.model_selection import test_with_cross_validate

from metabci.brainda.algorithms.utils.visualization import confusion_matrix_disply
from metabci.brainda.algorithms.utils.visualization import plot_tsne_feature
from metabci.brainda.algorithms.utils.visualization import convWeight_to_waveform
from metabci.brainda.algorithms.utils.visualization import convWeight_to_topography
from metabci.brainda.algorithms.utils.visualization import attentionWeight_Visualization

from metabci.brainda.paradigms import MotorImagery
from metabci.brainda.datasets.ustb2025mi4c import USTB2025MI4C
from metabci.brainflow.amplifiers import OpenBCIInputParams, OpenBCIShim, BoardIds

from datasets import MetaBCIData
from mne.filter import resample
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')



# 对raw操作,例如滤波
def raw_hook(raw, caches):
    # do something with raw object
    raw.filter(6, 30, l_trans_bandwidth=2, h_trans_bandwidth=5,
               phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches


# 按照0,1,2,...重新排列标签
def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


# 带通滤波
def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2*freq0/srate
    wn2 = 2*freq1/srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new

# 读取官方数据
def get_data(srate, subjects, pick_chs):
    # 截取数据的时间段
    stim_interval = [(0, 4)]

    # //.datasets.py中按照metabci.brainda.datasets数据结构自定义数据类MetaBCIData
    dataset = MetaBCIData(
        subjects=subjects, srate=srate,
        paradigm='imagery', pattern='imagery')  # declare the dataset
    paradigm = MotorImagery(
        channels=dataset.channels,
        events=dataset.events,
        intervals=stim_interval,
        srate=srate)
    paradigm.register_raw_hook(raw_hook)
    X, y, meta = paradigm.get_data(
        dataset,
        subjects=subjects,
        return_concat=True,
        n_jobs=2,
        verbose=False)
    y = label_encoder(y, np.unique(y))
    print("Loding data successfully")

    # 打印数据信息以便调试
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"实际使用的通道数: {X.shape[1]}")
    print(f"预定义通道列表长度: {len(pick_chs)}")

    return X, y, meta


# 读取自采数据
def get_data_new(subjects):
    id_to_label = {'1': 0, '2': 1, '3': 2, '4': 3}
    subject = subjects[0]
    ustb2025mi4c_path = r"data\ustb2025mi4c"

    train_feature_list, train_label_list = [], []
    for run in range(1, 5):
        feature_file = os.path.join(ustb2025mi4c_path, "S0{:d}_run{:d}.set".format(subject, run))
        epochs = mne.read_epochs_eeglab(feature_file)
        if subject == 1:
            label_run = epochs.events[:, -1]-1+run-1
        else:
            events = epochs.events[:, -1]
            event_id = [key for value in events for key, val in epochs.event_id.items() if val == value]
            label_run = [val for id in event_id for key, val in id_to_label.items() if key == id]
        epoch_data = epochs.get_data()  # 形状：(n_epochs, n_channels, n_times)

        # 滑窗分割：窗口长度256，步长256
        n_epochs, n_channels, n_times = epoch_data.shape
        window_length = 256
        stride = 256
        n_segments = n_times // stride  # 应为4（1024/256）
        segmented_data = []
        label_run_expanded = []
        for i in range(n_epochs):
            for j in range(n_segments):
                start = j * stride
                end = start + window_length
                segment = epoch_data[i, :, start:end]  # 提取窗口数据
                segmented_data.append(segment)
                label_run_expanded.append(label_run[i])  # 复制对应标签
        segmented_data = np.array(segmented_data)  # 形状：(n_epochs * n_segments, n_channels, window_length)
        label_run_expanded = np.array(label_run_expanded)  # 形状：(n_epochs * n_segments,)
        train_feature_list.append(segmented_data)
        train_label_list.append(label_run_expanded)

        # train_feature_list.append(epoch_data)
        # train_label_list.append(label_run)

    train_feature_arr = np.array(train_feature_list)
    train_label_arr = np.array(train_label_list)
    train_feature_arr = np.concatenate(train_feature_arr, axis=0)
    train_label_arr = np.concatenate(train_label_arr, axis=0)
    scaler = StandardScaler() # 标准化数据（按通道归一化）
    data_reshaped = train_feature_arr.reshape(train_feature_arr.shape[0], -1)  # 展平为 [n_epochs, n_channels * n_times]
    data_normalized = scaler.fit_transform(data_reshaped)
    train_feature_arr = data_normalized.reshape(train_feature_arr.shape)  # 恢复形状

    train_feature, test_feature, train_label, test_label = train_test_split(train_feature_arr, train_label_arr, test_size=0.2, shuffle=True, random_state=20250702, stratify=train_label_arr)

    return train_feature, train_label, test_feature, test_label

# 训练模型
def train_model(X, y, srate=1000):
    y = np.reshape(y, (-1))
    # 降采样
    X = resample(X, up=256, down=srate)
    # 滤波
    # X = bandpass(X, 6, 30, 256)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # brainda.algorithms.decomposition.csp.MultiCSP
    wp = [(4, 8), (8, 12), (12, 30)]
    ws = [(2, 10), (6, 14), (10, 32)]
    filterbank = generate_filterbank(wp, ws, srate=256, order=4, rp=0.5)
    # model = make_pipeline(
    #     MultiCSP(n_components = 2),
    #     LinearDiscriminantAnalysis())
    # model = make_pipeline(*[
    #     FBCSP(n_components=5,
    #           n_mutualinfo_components=4,
    #           filterbank=filterbank),
    #     SVC()
    # ])
    # model = EEGNet(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    # model = TCNet_Fusion(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    # model = EEG_Conformer(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    model = EEG_TCNet(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    # model = ATCNet(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    # fit()训练模型
    y = y.astype(np.int64)
    model = model.fit(X, y)

    return model


# 预测标签
def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # 降采样
    X = resample(X, up=256, down=srate)
    # 滤波
    X = bandpass(X, 8, 30, 256)
    # 零均值单位方差 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # predict()预测标签
    p_labels = model.predict(X)
    return p_labels


# 计算离线正确率
def offline_validation(X, y, srate=1000):
    y = np.reshape(y, (-1))

    kfold_accs = []
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)  # 留一法交叉验证
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        model = train_model(X_train, y_train, srate=srate)  # 训练模型
        p_labels = model_predict(X_test, srate=srate, model=model)  # 预测标签
        kfold_accs.append(np.mean(p_labels == y_test))  # 记录正确率

    return np.mean(kfold_accs)


def offline_validation_new(X, y, subject, X_test=None, y_test=None):
    ###============================ Use the GPU to train ============================###
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print("current device:", torch.cuda.get_device_name(device))
    ###============================ Sets the seed for random numbers ============================###
    seed_value = 20250702  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    ###============ model ============###
    # model = EEGNet(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    # model = TCNet_Fusion(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    model = EEG_Conformer(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    # model = EEG_TCNet(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    # model = ATCNet(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    model_name = "EEG_Conformer"
    model = model.module.to(dtype=torch.float32).to(device)
    ###============ criterion ============###
    criterion = nn.CrossEntropyLoss().to(device)
    ###============ optimizer ============###
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.7, 0.999), weight_decay=0.001) #
    ###============================ Initialization parameters ============================###
    lr_scheduler = None
    frist_epochs = 1000
    eary_stop_epoch = 100
    second_epochs = 100
    batch_size = 64
    kfolds = 5
    model_savePath = "checkpoints\\{}\\Subject_0{}".format(model_name+"_ustb2025mi4c", str(subject[0])) # +"_ustb2025mi4c"
    if not os.path.exists(model_savePath):
        os.makedirs(model_savePath)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20250702, stratify=y)
    X_train, y_train = X, y

    model_training_two_stage(model, criterion, optimizer, lr_scheduler, frist_epochs, eary_stop_epoch, second_epochs, batch_size,
                             X_train, y_train, kfolds, device,
                             model_name, subject, model_savePath)
    # model_training_two_stage_up(model, criterion, optimizer, lr_scheduler, frist_epochs, eary_stop_epoch, second_epochs, batch_size,
    #                             X_train, y_train, kfolds, device,
    #                             model_name, subject, model_savePath)
    features_path = "checkpoints\\{}\\Subject_0{}\\visualization\\tsne_feature\\".format(model_name+"_ustb2025mi4c", str(subject[0])) # +"_ustb2025mi4c"
    if not os.path.exists(features_path):
        os.makedirs(features_path)
    test_with_cross_validate(model, device, X_test, y_test, model_savePath, kfolds, subject, visual=False, features_path=features_path)


def model_visualization(model_name, y_test, channels, subject):
    ###============================ Initialization parameters ============================###
    output_dir = "checkpoints\\{}\\Subject_0{}\\visualization\\".format(model_name, str(subject[0]))
    ###============================ Confusion Matrix ============================###
    labels = ['Left', 'Right']
    confusion = np.array([[0.83, 0.17], [0.00, 1.00]])
    confusion_matrix_disply(confusion, labels, output_dir, dpi=300)
    ###============================ featre_tSNE ============================###
    true_labels = y_test
    features_path = "checkpoints\\{}\\Subject_0{}\\visualization\\tsne_feature\\embed_feature.pkl".format(model_name, str(subject[0]))
    plot_tsne_feature(features_path, true_labels, output_dir, file_name='embed_tSNE', dpi=300, random_state=44)
    features_path = "checkpoints\\{}\\Subject_0{}\\visualization\\tsne_feature\\transformer_feature.pkl".format(model_name, str(subject[0]))
    plot_tsne_feature(features_path, true_labels, output_dir, file_name='transformer_tSNE', dpi=300, random_state=44)
    ###============================ Weight Visualization ============================###
    model_savePath = "checkpoints\\{}\\Subject_0{}\\{}_sub[1]_fold1_acc1.0.pth".format(model_name, str(subject[0]), model_name)
    model_convLayerName = "model.patch_embed.conv1.weight"
    convWeight_to_waveform(model_savePath, model_convLayerName, output_dir, ylim=1, scalingCol=3, scalingRow=2, dpi=300)
    model_convLayerName = "model.patch_embed.conv2.weight"
    convWeight_to_topography(model_savePath, model_convLayerName, output_dir, scalingCol=3, scalingRow=2, dpi=300, channelsName=channels)
    attention_savePath = "checkpoints\\{}\\Subject_0{}\\visualization\\attention_feature\\attention_feature.pkl".format(model_name, str(subject[0]))
    attentionWeight_Visualization(attention_savePath, output_dir, scalingCol=3, scalingRow=2, dpi=300)


# 频域分析
def frequency_feature(X, meta, event, srate=1000):
    # brainda.algorithms.feature_analysis.freq_analysis.FrequencyAnalysis
    Feature_R = FrequencyAnalysis(X, meta, event=event, srate=srate)

    # 计算模板信号,调用FrequencyAnalysis.stacking_average()
    mean_data = Feature_R.stacking_average(data=[], _axis=0)

    plt.subplot(121)
    # 计算PSD值,调用FrequencyAnalysis.power_spectrum_periodogram()
    f, den = Feature_R.power_spectrum_periodogram(mean_data[8])  # C3
    plt.plot(f, den * 5)
    plt.xlim(0, 35)
    plt.ylim(0, 0.3)
    plt.title("right_hand :C3")
    plt.ylabel('PSD [V**2]')
    plt.xlabel('Frequency [Hz]')
    # plt.show()

    plt.subplot(122)
    # 计算PSD值,调用FrequencyAnalysis.power_spectrum_periodogram()
    f, den = Feature_R.power_spectrum_periodogram(mean_data[8])  # C4
    plt.plot(f, den * 5)
    plt.xlim(0, 35)
    plt.ylim(0, 0.3)
    plt.title("right_hand :C4")
    plt.ylabel('PSD [V**2]')
    plt.xlabel('Frequency [Hz]')
    plt.show()


# 时频分析
def time_frequency_feature(X, y, srate=1000, actual_channels=None):
    # brainda.algorithms.feature_analysis.time_freq_analysis.TimeFrequencyAnalysis
    TimeFreq_Process = TimeFrequencyAnalysis(srate)

    # 短时傅里叶变换  左手握拳
    index_8hz = np.where(y == 0)  # y=0 左手
    data_8hz = np.squeeze(X[index_8hz, :, :])
    mean_data_8hz = np.mean(data_8hz, axis=0)
    nfft = mean_data_8hz.shape[1]
    # 调用TimeFrequencyAnalysis.fun_stft()
    f, t, Zxx1 = TimeFreq_Process.fun_stft(
        mean_data_8hz, nperseg=1000, axis=1, nfft=nfft)
    Zxx_Pz1 = Zxx1[8, :, :]  # 导联选择4：C3
    Zxx_Pz2 = Zxx1[8, :, :]  # 导联选择6：C4
    # 时频图
    plt.subplot(321)
    plt.pcolormesh(t, f, np.abs(Zxx_Pz1))
    plt.ylim(0, 45)
    plt.title('STFT Left C3')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()
    plt.subplot(322)
    plt.pcolormesh(t, f, np.abs(Zxx_Pz2))
    plt.ylim(0, 45)
    plt.title('STFT Left C4')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()

    # 短时傅里叶变换  右手握拳
    index_8hz2 = np.where(y == 1)  # y=0 右手
    data_8hz2 = np.squeeze(X[index_8hz2, :, :])
    mean_data_8hz2 = np.mean(data_8hz2, axis=0)
    nfft = mean_data_8hz2.shape[1]
    # 调用TimeFrequencyAnalysis.fun_stft()
    f, t, Zxx2 = TimeFreq_Process.fun_stft(
        mean_data_8hz2, nperseg=1000, axis=1, nfft=nfft)
    Zxx_Pz3 = Zxx2[8, :, :]  # 导联选择3：C3
    Zxx_Pz4 = Zxx2[8, :, :]  # 导联选择5：C4
    # 时频图
    plt.subplot(323)
    plt.pcolormesh(t, f, np.abs(Zxx_Pz3))
    plt.ylim(0, 45)
    plt.title('STFT Right C3')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()
    plt.subplot(324)
    plt.pcolormesh(t, f, np.abs(Zxx_Pz4))
    plt.ylim(0, 45)
    plt.title('STFT Right C4')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar()
    plt.show()

    # 脑地形图 - 修复调用方式
    temp_map = np.mean(Zxx1, axis=1)
    temp = np.mean(temp_map, axis=1)
    # 使用实际的通道名称
    if actual_channels is not None:
        try:
            TimeFreq_Process.fun_topoplot(np.abs(temp), actual_channels)
            print("左手脑地形图绘制成功")
        except Exception as e:
            print(f"左手脑地形图绘制失败:{e}")
    else:
        print("跳过左手脑地形图绘制 - 缺少通道信息")

    # topomap
    temp_map = np.mean(Zxx2, axis=1)
    temp = np.mean(temp_map, axis=1)
    # 使用实际的通道名称
    if actual_channels is not None:
        try:
            TimeFreq_Process.fun_topoplot(np.abs(temp), actual_channels)
            print("右手脑地形图绘制成功")
        except Exception as e:
            print(f"右手脑地形图绘制失败: {e}")
    else:
        print("跳过右手脑地形图绘制 - 缺少通道信息")


# 完整的熵特征分析函数 - 验证所有熵分析方法
def entropy_feature(X, y, srate=1000, ch_names=None):
    """
    完整的熵特征分析，测试所有可用的熵计算方法
    """
    print("=" * 60)
    print("开始熵特征分析 - 验证所有熵分析函数")
    print("=" * 60)

    # 初始化熵分析器
    EntropyAnalyzer = EntropyAnalysis(srate)

    # 获取实际的通道数量和通道名称
    n_channels = X.shape[1]
    if ch_names is None:
        ch_names = [f"Ch{i}" for i in range(n_channels)]
    else:
        # 确保通道名称数量与实际数据通道数量匹配
        if len(ch_names) != n_channels:
            print(f"警告: 通道名称数量({len(ch_names)})与数据通道数量({n_channels})不匹配")
            print(f"使用前{n_channels}个通道名称")
            ch_names = ch_names[:n_channels]

    # 取第一个试次作为示例进行分析
    sample_trial = X[0]  # shape: (n_channels, n_times)
    print(f"示例数据形状: {sample_trial.shape}")
    print(f"通道数: {sample_trial.shape[0]}, 时间点数: {sample_trial.shape[1]}")
    print(f"使用的通道名称: {ch_names}")

    # 1. 差分熵 (Differential Entropy)
    print("\n1. 计算差分熵 (Differential Entropy)...")
    try:
        de_values = EntropyAnalyzer.differential_entropy(sample_trial)
        print(f"   差分熵计算成功，形状: {de_values.shape}")
        print(f"   差分熵值范围: [{np.min(de_values):.4f}, {np.max(de_values):.4f}]")
        print(f"   差分熵均值: {np.mean(de_values):.4f}")

        # 可视化差分熵脑地形图
        print("   绘制差分熵脑地形图...")
        try:
            EntropyAnalyzer.fun_topoplot(de_values, ch_names=ch_names)
            print("   差分熵脑地形图绘制成功")
        except Exception as e:
            print(f"   差分熵脑地形图绘制失败: {e}")

    except Exception as e:
        print(f"   差分熵计算失败: {e}")

    # 2. 样本熵 (Sample Entropy)
    print("\n2. 计算样本熵 (Sample Entropy)...")
    try:
        sampen_values = EntropyAnalyzer.sample_entropy(sample_trial, m=2, r=None)
        print(f"   样本熵计算成功，形状: {sampen_values.shape}")
        print(f"   样本熵值范围: [{np.min(sampen_values):.4f}, {np.max(sampen_values):.4f}]")
        print(f"   样本熵均值: {np.mean(sampen_values):.4f}")

        # 可视化样本熵脑地形图
        print("   绘制样本熵脑地形图...")
        try:
            EntropyAnalyzer.fun_topoplot(sampen_values, ch_names=ch_names)
            print("   样本熵脑地形图绘制成功")
        except Exception as e:
            print(f"   样本熵脑地形图绘制失败: {e}")

    except Exception as e:
        print(f"   样本熵计算失败: {e}")

    # 3. 近似熵 (Approximate Entropy)
    print("\n3. 计算近似熵 (Approximate Entropy)...")
    try:
        apen_values = EntropyAnalyzer.approximate_entropy(sample_trial, m=2, r=None)
        print(f"   近似熵计算成功，形状: {apen_values.shape}")
        print(f"   近似熵值范围: [{np.min(apen_values):.4f}, {np.max(apen_values):.4f}]")
        print(f"   近似熵均值: {np.mean(apen_values):.4f}")

        # 可视化近似熵脑地形图
        print("   绘制近似熵脑地形图...")
        try:
            EntropyAnalyzer.fun_topoplot(apen_values, ch_names=ch_names)
            print("   近似熵脑地形图绘制成功")
        except Exception as e:
            print(f"   近似熵脑地形图绘制失败: {e}")

    except Exception as e:
        print(f"   近似熵计算失败: {e}")

    # 4. 模糊熵 (Fuzzy Entropy)
    print("\n4. 计算模糊熵 (Fuzzy Entropy)...")
    try:
        fuzzyen_values = EntropyAnalyzer.fuzzy_entropy(sample_trial, m=2, r=None)
        print(f"   模糊熵计算成功，形状: {fuzzyen_values.shape}")
        print(f"   模糊熵值范围: [{np.min(fuzzyen_values):.4f}, {np.max(fuzzyen_values):.4f}]")
        print(f"   模糊熵均值: {np.mean(fuzzyen_values):.4f}")

        # 可视化模糊熵脑地形图
        print("   绘制模糊熵脑地形图...")
        try:
            EntropyAnalyzer.fun_topoplot(fuzzyen_values, ch_names=ch_names)
            print("   模糊熵脑地形图绘制成功")
        except Exception as e:
            print(f"   模糊熵脑地形图绘制失败: {e}")

    except Exception as e:
        print(f"   模糊熵计算失败: {e}")

    # 5. 排列熵 (Permutation Entropy)
    print("\n5. 计算排列熵 (Permutation Entropy)...")
    try:
        pe_values = EntropyAnalyzer.permutation_entropy(sample_trial, m=3, delay=1)
        print(f"   排列熵计算成功，形状: {pe_values.shape}")
        print(f"   排列熵值范围: [{np.min(pe_values):.4f}, {np.max(pe_values):.4f}]")
        print(f"   排列熵均值: {np.mean(pe_values):.4f}")

        # 可视化排列熵脑地形图
        print("   绘制排列熵脑地形图...")
        try:
            EntropyAnalyzer.fun_topoplot(pe_values, ch_names=ch_names)
            print("   排列熵脑地形图绘制成功")
        except Exception as e:
            print(f"   排列熵脑地形图绘制失败: {e}")

    except Exception as e:
        print(f"   排列熵计算失败: {e}")

    # 6. 比较不同条件下的熵值
    print("\n6. 比较不同运动想象条件下的熵特征...")
    try:
        # 分别计算左手和右手条件下的差分熵
        left_indices = np.where(y == 0)[0]
        right_indices = np.where(y == 1)[0]

        if len(left_indices) > 0 and len(right_indices) > 0:
            left_data = X[left_indices]
            right_data = X[right_indices]

            # 计算平均差分熵
            left_de = np.mean([EntropyAnalyzer.differential_entropy(trial) for trial in left_data], axis=0)
            right_de = np.mean([EntropyAnalyzer.differential_entropy(trial) for trial in right_data], axis=0)

            print(f"   左手运动想象平均差分熵: {np.mean(left_de):.4f}")
            print(f"   右手运动想象平均差分熵: {np.mean(right_de):.4f}")
            print(f"   差异 (右-左): {np.mean(right_de - left_de):.4f}")

            # 可视化差异脑地形图
            print("   绘制左右手差异脑地形图...")
            try:
                EntropyAnalyzer.fun_topoplot(right_de - left_de, ch_names=ch_names)
                print("   左右手差异脑地形图绘制成功")
            except Exception as e:
                print(f"   左右手差异脑地形图绘制失败: {e}")

    except Exception as e:
        print(f"   条件比较失败: {e}")

    # 7. 批量处理所有试次的熵特征
    print("\n7. 批量计算所有试次的熵特征...")
    try:
        all_de_values = np.array([EntropyAnalyzer.differential_entropy(trial) for trial in X])
        print(f"   所有试次差分熵形状: {all_de_values.shape}")
        print(f"   跨试次差分熵均值: {np.mean(all_de_values, axis=0)}")
        print(f"   跨试次差分熵标准差: {np.std(all_de_values, axis=0)}")

        # 可视化跨试次平均熵值
        mean_entropy_across_trials = np.mean(all_de_values, axis=0)
        print("   绘制跨试次平均熵值脑地形图...")
        try:
            EntropyAnalyzer.fun_topoplot(mean_entropy_across_trials, ch_names=ch_names)
            print("   跨试次平均熵值脑地形图绘制成功")
        except Exception as e:
            print(f"   跨试次平均熵值脑地形图绘制失败: {e}")

    except Exception as e:
        print(f"   批量处理失败: {e}")

    print("\n" + "=" * 60)
    print("熵特征分析完成 - 所有函数验证结束")
    print("=" * 60)


if __name__ == '__main__':

    ### =========================== 新增ustb2025mi4c数据加载功能测试 ============================ ###
    # dataset = USTB2025MI4C()
    # data = dataset._get_single_subject_data(subject=1)  # 读取被试1的数据
    # print(data)

    ### =========================== 新增脑电解码模型在fit模式下训练与推理测试 ============================ ###
    # srate = 256
    # subjects = [1]
    # pick_chs = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2',
    #             'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    #             'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5',
    #             'P3', 'P1', 'Pz', 'P2', 'P4', 'P6']
    # X, y, meta = get_data(srate=srate, subjects=subjects, pick_chs=pick_chs)
    # acc = offline_validation(X, y, srate=srate)  # 计算离线准确率
    # print("Current Model accuracy:", acc)

    ### =========================== 新增脑电解码模型在新增两种模型训练方式的训练与推理测试 ============================ ###
    subjects = [1]
    X, y, test_feature_arr, test_label_arr = get_data_new(subjects)
    offline_validation_new(X, y, subject=subjects, X_test=test_feature_arr, y_test=test_label_arr)  # 计算离线准确率 , X_test=test_feature_arr, y_test=test_label_arr

    ### =========================== 新增的模型可视化方法测试 ============================ ###
    # srate = 256
    # subjects = [1]
    # pick_chs = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2',
    #             'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    #             'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5',
    #             'P3', 'P1', 'Pz', 'P2', 'P4', 'P6']
    # X, y, meta = get_data(srate=srate, subjects=subjects, pick_chs=pick_chs)
    # actual_n_channels = X.shape[1]
    # if len(pick_chs) >= actual_n_channels:
    #     actual_channels = pick_chs[:actual_n_channels]
    # else: # 如果预定义通道数不够，添加额外的通道名
    #     actual_channels = pick_chs + [f"Extra_Ch{i}" for i in range(len(pick_chs), actual_n_channels)]
    # print(f"实际使用的通道名称: {actual_channels}")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20250702, stratify=y)
    # model_visualization(model_name="EEG_Conformer", y_test=y_test, channels=actual_channels, subject=subjects)

    ### =========================== 新增熵特征提取方法测试 ============================ ###
    # srate = 256
    # subjects = [1]
    # pick_chs = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2',
    #             'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    #             'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5',
    #             'P3', 'P1', 'Pz', 'P2', 'P4', 'P6']
    # X, y, meta = get_data(srate=srate, subjects=subjects, pick_chs=pick_chs)
    # actual_n_channels = X.shape[1]
    # if len(pick_chs) >= actual_n_channels:
    #     actual_channels = pick_chs[:actual_n_channels]
    # else: # 如果预定义通道数不够，添加额外的通道名
    #     actual_channels = pick_chs + [f"Extra_Ch{i}" for i in range(len(pick_chs), actual_n_channels)]
    # print(f"实际使用的通道名称: {actual_channels}")
    # entropy_feature(X, y, srate, ch_names=actual_channels)

    ### =========================== 新增OpenBCI设备支持代码的测试 ============================ ###
    # params = OpenBCIInputParams()
    # params.serial_port = "COM4"  # 开发板OpenBCI串口
    # board = OpenBCIShim(BoardIds.CYTON_DAISY_BOARD.value, params)  # 在子进程中启用OpenBCI
    # board.prepare_session()
    # board.start_stream()  # 打开OpenBCI的数据流
    # while True:
    #     data = board.get_current_board_data(num_samples=125)
    #     print(data)

    ### =========================== 新增新增VR游戏通信控制模块测试 ============================ ###
    # SOCKET_HOST = "localhost"  # Socket 目标 IP
    # SOCKET_PORT = 12345  # Socket 目标端口
    # server_socket, client_socket = command_output(SOCKET_HOST, SOCKET_PORT)
    # while True:
    #     command = input("Enter command (1, 2, 3, 4) or 'exit' to quit: ")
    # 
    #     if command.lower() == 'exit':
    #         break
    #     if command in ['1', '2', '3', '4']:
    #         message = command.encode('ascii')  # 转换为字节
    #         client_socket.send(message)
    #         print(f"Sent: {command}")
    #     else:
    #         print("Invalid command! Use 1, 2, 3, 4, or exit.")
    # 
    #     # 接收客户端的到达指令
    #     try:
    #         socket_data = client_socket.recv(1024).decode('ascii')
    #         if socket_data:
    #             print(f"Received from client: {socket_data}")
    #     except:
    #         print("Error receiving data")
