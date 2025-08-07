# import pickle
# import numpy as np
# import mne
#
#
# def ESI_trial(trial, sfreq=500):
#     eeg = trial['eeg']
#     info = mne.create_info([f'EEG {i}' for i in range(eeg.shape[0])], sfreq, ['eeg'] * eeg.shape[0])
#     raw = mne.io.RawArray(eeg, info)
#     raw.set_eeg_reference('average')
#     raw.filter(0.1, 40)
#
#     montage = mne.channels.make_standard_montage('standard_1020')
#     raw.set_montage(montage)
#
#     subject = 'fsaverage'
#     subjects_dir = mne.datasets.fetch_fsaverage(verbose=True)
#     src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir)
#     model = mne.make_bem_model(subject=subject, ico=4, conductivity=[0.3], subjects_dir=subjects_dir)
#     bem = mne.make_bem_solution(model)
#     trans = 'fsaverage'
#
#     fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True)
#
#     events = np.array([[0, 0, 1]])
#     event_id = dict(grasp=1)
#     epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=eeg.shape[1]/sfreq - 1/sfreq, preload=True)
#     evoked = epochs.average()
#     noise_cov = mne.compute_covariance(epochs, tmax=0.0)
#     inv_op = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)
#     stc = mne.minimum_norm.apply_inverse(evoked, inv_op, lambda2=1. / 9., method='sLORETA')
#     return stc.data  # shape: (n_sources, n_times)
#
#
#
# def ESI_structure():
#     # 1.load data from .pkl
#     with open('path_to_trial.pkl', 'rb') as f:
#         trials = pickle.load(f)
#
#     trial = trials[0]  # 任选一个 trial 测试
#     eeg_data = trial['eeg']  # shape: (n_channels, n_times)
#
#     # 2.build MNE RawArray object
#     n_channels = eeg_data.shape[0]
#     sfreq = 500  # 采样率，按你的数据设定
#     ch_names = [f'EEG {i}' for i in range(n_channels)]
#     ch_types = ['eeg'] * n_channels
#
#     info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
#     raw = mne.io.RawArray(eeg_data, info)
#
#     # 可选：去均值、带通滤波等
#     raw.set_eeg_reference('average')
#     raw.filter(0.1, 40)
#
#     # 3.head model
#     subjects_dir = mne.datasets.fetch_fsaverage(verbose=True)
#     subject = 'fsaverage'
#
#     src = mne.setup_source_space(subject, spacing='oct6', add_dist=False, subjects_dir=subjects_dir)
#
#     model = mne.make_bem_model(subject=subject, ico=4, conductivity=[0.3], subjects_dir=subjects_dir)
#     bem = mne.make_bem_solution(model)
#
#     montage = mne.channels.make_standard_montage('standard_1020')  # 或者根据你电极布局自定义
#     raw.set_montage(montage)
#
#     trans = 'fsaverage'  # 默认头模转换
#     fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True)
#
#     # 4.create events and create epoch
#     # 创建一个 epoch 时间窗口（例如 -0.2s 到 0.5s）
#     events = np.array([[0, 0, 1]])  # 一个事件，index 0
#     event_id = dict(grasp=1)
#
#     epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=eeg_data.shape[1] / sfreq - 1 / sfreq,
#                         baseline=None, preload=True)
#
#     evoked = epochs.average()
#
#     # 5.compute cov inverse get esi
#     noise_cov = mne.compute_covariance(epochs, tmax=0.0)
#
#     inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)
#
#     stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator,
#                                          lambda2=1.0 / 9.0, method='sLORETA')
#
#     # 得到源域特征：stc.data.shape = (n_sources, n_times)


import numpy as np
import matplotlib.pyplot as plt

def plot_frequency_response(kernel):
    # 计算频率响应（FFT）
    freq_resp = np.fft.fft(kernel, 512)
    freq_resp = np.abs(freq_resp)
    freq_resp = freq_resp / np.max(freq_resp)  # 归一化
    freqs = np.linspace(0, 0.5, len(freq_resp)//2)  # 采样频率归一化，0~0.5为Nyquist

    plt.plot(freqs, freq_resp[:len(freq_resp)//2])
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Magnitude')
    plt.ylim(0, 1.1)
    plt.grid(True)

# 构造两个卷积核：一个大核，一个小核
# 这里用简单的矩形窗口表示滤波器权重，方便观察频率响应差异

large_kernel = np.ones(15) / 15  # 大核卷积核，长度15
small_kernel = np.ones(3) / 3    # 小核卷积核，长度3

plt.figure(figsize=(10, 5))
plot_frequency_response(large_kernel)
plt.title('Frequency Response of Large Kernel (length=15)')
plt.show()

plt.figure(figsize=(10, 5))
plot_frequency_response(small_kernel)
plt.title('Frequency Response of Small Kernel (length=3)')
plt.show()

# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
#
# # 模拟输入信号：64通道，256时间点
# eeg_input = torch.randn(1, 1, 64, 256)
#
# # 定义卷积层
# conv = nn.Conv2d(1, 16, kernel_size=(1,63), stride=1, padding=(0,31), bias=False)
#
# # 获取第一个卷积核的权重（shape: [1, 63]）
# kernel = conv.weight[0, 0, 0, :].detach().cpu().numpy()
#
# # 看其中一个通道（比如第32个通道）和它卷积前后的对比
# channel_32 = eeg_input[0, 0, 31, :].detach().cpu().numpy()
#
# # 卷积操作（仅对一个通道 + 一个核，模拟）
# import numpy as np
# from scipy.signal import convolve
#
# conv_output = convolve(channel_32, kernel[::-1], mode='same')  # 卷积核翻转，匹配数学卷积
#
# # 画图
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(channel_32)
# plt.title("原始EEG通道信号（第32通道）")
# plt.subplot(1, 2, 2)
# plt.plot(conv_output)
# plt.title("卷积后的响应（第1个卷积核）")
# plt.tight_layout()
# plt.show()


