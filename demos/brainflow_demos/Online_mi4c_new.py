# -*- coding: utf-8 -*-
# License: MIT License
"""
MI offline analysis.

"""
import os
import torch
from sklearn.model_selection import train_test_split
from metabci.brainda.algorithms.deep_learning import EEG_Conformer
from metabci.brainda.algorithms.utils.model_selection import test_with_cross_validate
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler
import warnings
import pylsl
import time
from metabci.brainflow.workers import command_output
from torch.nn.functional import softmax
import keyboard
warnings.filterwarnings('ignore')


# ===== 参数配置 =====
LSL_EEG_STREAM_NAME = "CURRYStream"  # LSL EEG 流的名称（需与 NeuroScan 发送的名称匹配）
MODEL_PATH = r"F:\2025世界机器人大赛-BCI脑控机器人大赛\MetaBCI-master\demos\brainflow_demos\checkpoints\EEG_Conformer_ustb2025mi4c_CPU\Subject_01\EEG_Conformer_sub[1]_fold2_acc1.0.pth"  # 预训练模型路径
SOCKET_HOST = "192.168.3.17"  # Socket 目标 IP
SOCKET_PORT = 12345  # Socket 目标端口
TOTAL_CHANNELS = 35 # LSL原始流的通道总数
USED_CHANNELS = 32  # 实际使用的通道数（前32个）
REF_CHANNELS = [21, 22]   # 重参考通道索引（第22和23通道，Python从0开始）
EEG_SAMPLING_RATE = 256  # 采样率（Hz，需与 NeuroScan 设置一致）
WINDOW_LENGTH_SEC = 1  # 模型输入时间窗口（秒）
WINDOW_LENGTH_SAMPLES = EEG_SAMPLING_RATE * WINDOW_LENGTH_SEC  # 1秒对应的样本数
CLASSES_NUMBER = 4
PREDICTION_TO_LABEL = {0: 'left', 1: 'right', 2: 'retreat', 3: 'forward'}
FIRST_RUN = True
RUN_TIMES = 0
# 读取自采数据
def get_data_new(subjects):
    subject = subjects[0]
    ustb2025mi4c_path = r"data\ustb2025mi4c"

    train_feature_list, train_label_list = [], []
    for run in range(1, 5):
        feature_file = os.path.join(ustb2025mi4c_path, "S0{:d}_run{:d}.set".format(subject, run))
        epochs = mne.read_epochs_eeglab(feature_file)
        label_run = epochs.events[:, -1]-1+run-1
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


def offline_validation_new(X, y, subject, X_test=None, y_test=None):
    ###============================ Use the GPU to train ============================###
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print("current device:", device)
    ###============================ Sets the seed for random numbers ============================###
    seed_value = 20250702  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    ###============ model ============###
    model = EEG_Conformer(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=np.unique(y).size)
    model_name = "EEG_Conformer"
    model = model.module.to(dtype=torch.float32).to(device)
     ###============================ Initialization parameters ============================###
    kfolds = 5
    model_savePath = "checkpoints\\{}\\Subject_0{}".format(model_name+"_ustb2025mi4c_CPU", str(subject[0]))
    if not os.path.exists(model_savePath):
        os.makedirs(model_savePath)

    features_path = "checkpoints\\{}\\Subject_0{}\\visualization\\tsne_feature\\".format(model_name+"_ustb2025mi4c", str(subject[0]))
    if not os.path.exists(features_path):
        os.makedirs(features_path)
    test_with_cross_validate(model, device, X_test, y_test, model_savePath, kfolds, subject, visual=False, features_path=features_path)


if __name__ == "__main__":
    # ===== 初始化 =====
    # 1. 加载预训练模型
    subjects = [1]
    model = EEG_Conformer(n_channels=USED_CHANNELS-len(REF_CHANNELS), n_samples=EEG_SAMPLING_RATE, n_classes=CLASSES_NUMBER)
    model = model.module.to(dtype=torch.float32)
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 初始化 Socket
    server_socket, client_socket = command_output(SOCKET_HOST, SOCKET_PORT)

    # 3. 初始化 LSL EEG 数据流
    print(f"正在搜索 LSL EEG 流: {LSL_EEG_STREAM_NAME}...")
    streams = pylsl.resolve_stream('name', LSL_EEG_STREAM_NAME)
    if not streams:
        raise RuntimeError(f"未找到 EEG 流: {LSL_EEG_STREAM_NAME}")
    eeg_inlet = pylsl.StreamInlet(streams[0])
    print(f"已连接到 EEG 流: {eeg_inlet.info().name()}")

    # 4. 数据缓存队列（用于存储最近的 EEG 数据）
    eeg_buffer = np.zeros((USED_CHANNELS-len(REF_CHANNELS), WINDOW_LENGTH_SAMPLES))  # 形状: (通道数, 样本数)

    # 5. 加载离线数据，计算离线准确率
    X, y, test_feature_arr, test_label_arr = get_data_new(subjects)
    offline_validation_new(X, y, subject=subjects, X_test=test_feature_arr, y_test=test_label_arr)  # 计算离线准确率

    # 6. 加载离线数据，计算离线准确率
    print("按空格键开始实时解码...")
    keyboard.wait('space')

    try:
        print("开始实时解码（按 Ctrl+C 停止）...")
        while True:
            # 1. 从 LSL 获取 EEG 数据块（chunk）
            samples, timestamps = eeg_inlet.pull_chunk(timeout=0.1)  # 超时 100ms
            if samples:
                samples = np.array(samples).T  # 转置为 (通道数, 样本数)
                samples = samples[:USED_CHANNELS, :]  # 仅保留前32个通道 (32, N)

                # 2. 重参考预处理（减去第22和23通道的平均值）
                ref_signal = np.mean(samples[REF_CHANNELS, :], axis=0)  # 参考信号 (N,)
                samples_preprocessed = samples - ref_signal  # 所有32个通道重参考

                # 3. 移除参考通道（第22和23通道）
                mask = np.ones(USED_CHANNELS, dtype=bool)
                mask[REF_CHANNELS] = False  # 标记要删除的通道
                samples_cleaned = samples_preprocessed[mask, :]  # 形状: (30, N)

                # 4. 更新数据缓存（滑动窗口）
                new_samples = samples_cleaned.shape[1]
                if new_samples >= WINDOW_LENGTH_SAMPLES:
                    eeg_buffer = samples_cleaned[:, -WINDOW_LENGTH_SAMPLES:]  # 直接取最新 1s
                else:
                    eeg_buffer = np.roll(eeg_buffer, -new_samples, axis=1)  # 滑动窗口
                    eeg_buffer[:, -new_samples:] = samples_cleaned  # 填充新数据
                print('eeg_buffer shape: ', eeg_buffer.shape)

            # 检查是否满足解码条件
            decode_allowed = FIRST_RUN or (socket_data == 'arrived')
            if decode_allowed and eeg_buffer.shape[1] == WINDOW_LENGTH_SAMPLES:
                if FIRST_RUN:
                    FIRST_RUN = False
                # 执行解码（等待5秒期间持续更新数据）
                start_time = time.time()
                while time.time() - start_time < 5:
                    # 1. 从 LSL 获取 EEG 数据块（chunk）
                    samples, timestamps = eeg_inlet.pull_chunk(timeout=0.1)  # 超时 100ms
                    if samples:
                        samples = np.array(samples).T  # 转置为 (通道数, 样本数)
                        samples = samples[:USED_CHANNELS, :]  # 仅保留前32个通道 (32, N)
                        # 2. 重参考预处理（减去第22和23通道的平均值）
                        ref_signal = np.mean(samples[REF_CHANNELS, :], axis=0)  # 参考信号 (N,)
                        samples_preprocessed = samples - ref_signal  # 所有32个通道重参考
                        # 3. 移除参考通道（第22和23通道）
                        mask = np.ones(USED_CHANNELS, dtype=bool)
                        mask[REF_CHANNELS] = False  # 标记要删除的通道
                        samples_cleaned = samples_preprocessed[mask, :]  # 形状: (30, N)
                        # 4. 更新数据缓存（滑动窗口）
                        new_samples = samples_cleaned.shape[1]
                        if new_samples >= WINDOW_LENGTH_SAMPLES:
                            eeg_buffer = samples_cleaned[:, -WINDOW_LENGTH_SAMPLES:]  # 直接取最新 1s
                        else:
                            eeg_buffer = np.roll(eeg_buffer, -new_samples, axis=1)  # 滑动窗口
                            eeg_buffer[:, -new_samples:] = samples_cleaned  # 填充新数据
                    print('eeg_buffer shape: ', eeg_buffer.shape)
                    time.sleep(0.01)

                # 5秒后调用模型并发送结果
                input_data = eeg_buffer.reshape(1, *eeg_buffer.shape)  # 预处理（根据模型需求调整，如归一化、滤波等）添加 batch 维度
                if isinstance(input_data, torch.Tensor):
                    X_test = input_data.to(torch.float32)
                else:
                    X_test = torch.from_numpy(input_data).to(torch.float32)
                print('X_test shape: ', X_test.shape)
                prediction = model(X_test)  # 假设模型有 predict() 方法
                prediction = softmax(prediction, dim=-1).argmax(dim=-1).numpy()[0]
                print(f"解码结果: {PREDICTION_TO_LABEL[prediction-1]}")
                command = str(prediction)
                message = command.encode('ascii')  # 转换为字节
                client_socket.send(message)  # 发送结果
                # 接收客户端的到达指令
                try:
                    socket_data = client_socket.recv(1024).decode('ascii')
                    if socket_data:
                        print(f"Received from client: {socket_data}")
                except:
                    print("Error receiving data")
                RUN_TIMES = RUN_TIMES + 1
                if RUN_TIMES == 7: break

    except KeyboardInterrupt:
        print("用户终止程序...")
    finally:
        # 清理资源
        client_socket.close()
        server_socket.close()
        print("Server disconnected.")
