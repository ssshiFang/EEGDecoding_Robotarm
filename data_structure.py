import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.signal import butter, sosfiltfilt, resample_poly

# 加载数据
data = scipy.io.loadmat('D:/MyFolder/Msc_EEG/data5/WS_P5_S1.mat') # dict type

# 查看文件中的键
# print(data.keys()) #dict_keys(['__header__', '__version__', '__globals__', 'ws'])
# str str List (1,1) ws是一个包含一个元素de二维数组

my_data = data['ws'] # 数据文件存储在此
ws_obj = my_data[0, 0]

# print(type(ws_obj))     # 确认是否为 tuple/list 或 ndarray <class 'numpy.void'> numpy的一个结构化数组
# print(len(ws_obj))      # 看里面有多少元素:5
# print(ws_obj.dtype.names)     # ws中的字段为('name', 'participant', 'series', 'names', 'win')


#对于ws ('name', 'participant', 'series', 'names', 'win') 五个字段的访问

#ws_data = my_data[0, 0]
#ws各参数表示详细作用
if False:
    # 访问 name
    name = ws_data['name']
    print("name:", name) #['SN']

    # 访问 participant
    participant = ws_data['participant']
    print("participant:", participant) #participant: [[5]]

    # 访问 series（可能是主要数据）
    series = ws_data['series']
    print("series shape:", series.shape, "type:", type(series)) #series shape: (1, 1) type: <class 'numpy.ndarray'>

    # 访问 names（可能是通道名或传感器名）
    names = ws_data['names']
    print("names shape:", names.shape) #names shape: (1, 1)
    # (array([[array(['Fp1'], dtype='<U3'), array(['Fp2'], dtype='<U3'),
    #         array(['F7'], dtype='<U2'), array(['F3'], dtype='<U2'),
    #         array(['Fz'], dtype='<U2'), array(['F4'], dtype='<U2'),
    #         array(['F8'], dtype='<U2'), array(['FC5'], dtype='<U3'),
    #         array(['FC1'], dtype='<U3'), array(['FC2'], dtype='<U3'),
    #         array(['FC6'], dtype='<U3'), array(['T7'], dtype='<U2'),
    #         array(['C3'], dtype='<U2'), array(['Cz'], dtype='<U2'),
    #         array(['C4'], dtype='<U2'), array(['T8'], dtype='<U2'),
    #         array(['TP9'], dtype='<U3'), array(['CP5'], dtype='<U3'),
    #         array(['CP1'], dtype='<U3'), array(['CP2'], dtype='<U3'),
    #         array(['CP6'], dtype='<U3'), array(['TP10'], dtype='<U4'),
    #         array(['P7'], dtype='<U2'), array(['P3'], dtype='<U2'),
    #         array(['Pz'], dtype='<U2'), array(['P4'], dtype='<U2'),
    #         array(['P8'], dtype='<U2'), array(['PO9'], dtype='<U3'),
    #         array(['O1'], dtype='<U2'), array(['Oz'], dtype='<U2'),
    #         array(['O2'], dtype='<U2'), array(['PO10'], dtype='<U4')]],
    #       dtype=object), array([[array(['Ae1 - angle e sensor 1'], dtype='<U22'),

    #         array(['Ae2 - angle e sensor 2'], dtype='<U22'),
    #         array(['Ae3 - angle e sensor 3'], dtype='<U22'),
    #         array(['Ae4 - angle e sensor 4'], dtype='<U22'),
    #         array(['Ar1 - angle r sensor 1'], dtype='<U22'),
    #         array(['Ar2 - angle r sensor 2'], dtype='<U22'),
    #         array(['Ar3 - angle r sensor 3'], dtype='<U22'),
    #         array(['Ar4 - angle r sensor 4'], dtype='<U22'),
    #         array(['Az1 - angle z sensor 1'], dtype='<U22'),
    #         array(['Az2 - angle z sensor 2'], dtype='<U22'),
    #         array(['Az3 - angle z sensor 3'], dtype='<U22'),
    #         array(['Az4 - angle z sensor 4'], dtype='<U22'),
    #         array(['FX1 - force x plate 1'], dtype='<U21'),
    #         array(['FX2 - force x plate 2'], dtype='<U21'),
    #         array(['FY1 - force y plate 1'], dtype='<U21'),
    #         array(['FY2 - force y plate 2'], dtype='<U21'),
    #         array(['FZ1 - force z plate 1'], dtype='<U21'),
    #         array(['FZ2 - force z plate 2'], dtype='<U21'),
    #         array(['Px1 - position x sensor 1'], dtype='<U25'),
    #         array(['Px2 - position x sensor 2'], dtype='<U25'),
    #         array(['Px3 - position x sensor 3'], dtype='<U25'),
    #         array(['Px4 - position x sensor 4'], dtype='<U25'),
    #         array(['Py1 - position y sensor 1'], dtype='<U25'),
    #         array(['Py2 - position y sensor 2'], dtype='<U25'),
    #         array(['Py3 - position y sensor 3'], dtype='<U25'),
    #         array(['Py4 - position y sensor 4'], dtype='<U25'),
    #         array(['Pz1 - position z sensor 1'], dtype='<U25'),
    #         array(['Pz2 - position z sensor 2'], dtype='<U25'),
    #         array(['Pz3 - position z sensor 3'], dtype='<U25'),
    #         array(['Pz4 - position z sensor 4'], dtype='<U25'),
    #         array(['TX1 - torque x plate 1'], dtype='<U22'),
    #         array(['TX2 - torque x plate 2'], dtype='<U22'),
    #         array(['TY1 - torque y plate 1'], dtype='<U22'),
    #         array(['TY2 - torque y plate 2'], dtype='<U22'),
    #         array(['TZ1 - torque z plate 1'], dtype='<U22'),
    #         array(['TZ2 - torque z plate 2'], dtype='<U22'),
    #         array(['IndLF'], dtype='<U5'), array(['ThuLF'], dtype='<U5'),
    #         array(['LF'], dtype='<U2'), array(['IndGF'], dtype='<U5'),
    #         array(['ThuGF'], dtype='<U5'), array(['GF'], dtype='<U2'),
    #         array(['IndRatio'], dtype='<U8'),
    #         array(['ThuRatio'], dtype='<U8'),
    #         array(['GFLFRatio'], dtype='<U9')]], dtype=object), array([[array(['Anterior Deltoid'], dtype='<U16'),
    #         array(['Brachoradial'], dtype='<U12'),
    #         array(['Flexor Digitorum'], dtype='<U16'),
    #         array(['Common Extensor Digitorum'], dtype='<U25'),
    #         array(['First Dorsal Interosseus'], dtype='<U24')]], dtype=object))

    #是一个包含3个nadrray的组合数据
    #第一个：1x32的数组 第二个：1x47的数组 表示角度/力/位置/比例的传感器名 第三个：1x5的数组 表示EMG采集部位名

    # 访问 win（可能是窗口数据）
    win = ws_data['win']
    print("win shape:", win.shape) #win shape: (1, 28) 具有28个窗口


    series_obj = ws_data['names'][0, 0]
    print(type(series_obj))               # 预期是 np.void
    print(series_obj)        # 看看有哪些字段

    win_obj = ws_data['win']
    print(win_obj.shape)  # (1, 28) 确定有 28 个窗口

#获得传感器的位置信息
names_obj = ws_obj['names']
# print(names_obj[0,0].dtype.names) #('eeg', 'kin', 'emg')分别的位置信息
eeg_channel_names = names_obj['eeg']
#结果是
# raw_channel_data = [[array([[array(['Fp1'], dtype='<U3'), array(['Fp2'], dtype='<U3'),
#                               array(['F7'], dtype='<U2'), array(['F3'], dtype='<U2'),
#                               array(['Fz'], dtype='<U2'), array(['F4'], dtype='<U2'),
#                               array(['F8'], dtype='<U2'), array(['FC5'], dtype='<U3'),
#                               array(['FC1'], dtype='<U3'), array(['FC2'], dtype='<U3'),
#                               array(['FC6'], dtype='<U3'), array(['T7'], dtype='<U2'),
#                               array(['C3'], dtype='<U2'), array(['Cz'], dtype='<U2'),
#                               array(['C4'], dtype='<U2'), array(['T8'], dtype='<U2'),
#                               array(['TP9'], dtype='<U3'), array(['CP5'], dtype='<U3'),
#                               array(['CP1'], dtype='<U3'), array(['CP2'], dtype='<U3'),
#                               array(['CP6'], dtype='<U3'), array(['TP10'], dtype='<U4'),
#                               array(['P7'], dtype='<U2'), array(['P3'], dtype='<U2'),
#                               array(['Pz'], dtype='<U2'), array(['P4'], dtype='<U2'),
#                               array(['P8'], dtype='<U2'), array(['PO9'], dtype='<U3'),
#                               array(['O1'], dtype='<U2'), array(['Oz'], dtype='<U2'),
#                               array(['O2'], dtype='<U2'), array(['PO10'], dtype='<U4')]],
#                             dtype=object)]]

# 提取字符串
eeg_channel_ch_names = [ch[0] for ch in eeg_channel_names[0][0][0]]
#
# # 得到结果：
# print(eeg_channel_ch_names)
# # ['Fp1', 'Fp2', 'F7', 'F3', ..., 'PO10']  共 32 个通道


# 看第一个窗口内容
win_obj=ws_obj['win']
win = win_obj[0, 0]
# print(type(win))  # <class 'numpy.void'>
# print(win.dtype.names)  # ('eeg', 'kin', 'emg', 'eeg_t', 'emg_t', 'trial_start_time', 'LEDon', 'LEDoff', 'trial_end_time', 'weight', 'weight_id', 'surf', 'surf_id', 'weight_prev', 'weight_prev_id', 'surf_prev', 'surf_prev_id')
#
EEG = win_obj['eeg'] #(4907时间点, 32通道)
# # print(EEG)
# print(EEG[0,0].shape)
# print(len(EEG[0])) #28
# print(len(EEG[0,0]))

names_obj = ws_obj['names']
EMG = win_obj['emg']
# print(names_obj)
# print(EMG[0][0])
# print(EMG[0][0].shape)
# for i in range(0,27):
#     print(EMG[0][i].shape)

# EEG_t=win_obj['eeg_t'] #28
# print(EEG_t)
# print(len(EEG_t[0]))

# start=win_obj['trial_start_time']#28
# print(start)
# # print(len(start[0]))
# end=win_obj['trial_end_time']#28
# print(end)
# # print(len(end[0]))
#
# LED_on=win_obj['LEDon'] #全都是2
# print(LED_on)
# # print(len(LED_on[0])) #28
#
# LED_off=win_obj['LEDoff'] #6.5-7.5之间
# print(LED_off)
# # print(len(LED_off[0])) #28

#简单绘图
# # 示例：生成一个模拟的 EEG 数据 (32通道，4907时间点)
# # 如果你已经有真实数据，可以直接替换掉这个模拟数据
# eeg_data = EEG[0,0]
#
# # 数据是 eeg_data，shape 为 (4097, 32)
# # 我们先转置成 (32, 4097)
# eeg_data = eeg_data.T  # 转置之后，每一行就是一个通道
#
# # 参数
# n_channels, n_times = eeg_data.shape
# sampling_rate = 2000  # 根据实际采样率修改
# time = np.linspace(0, n_times / sampling_rate, n_times)
#
# # 画图
# plt.figure(figsize=(15, 10))
# offset = 2000  # 垂直偏移，防止通道重叠
#
# for i in range(n_channels):
#     plt.plot(time, eeg_data[i] + i * offset, label=f'Ch {i+1}')
#
# plt.title("EEG Signals (32 Channels)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude + Offset")
# plt.grid(True)
# plt.tight_layout()
# plt.legend(loc='upper right', fontsize=6)
# plt.show()




# ##
# win = ws_data['win']  # shape (1, 28) ，28个窗口
#
# n_trials = win.shape[1]
# X = []
# y = []  # 如果有标签的话，这里假设无，或你自行定义
#
# for i in range(n_trials):
#     trial = win[0, i]
#     eeg_data = trial['eeg']  # shape 可能是 (channels, timepoints)
#     eeg_t = trial['eeg_t']  # 时间戳（可以忽略）
#
#     # 转成 np.array
#     eeg_np = np.array(eeg_data)
#
#     # 确认形状（一般 EEGNet 输入是 (channels, timepoints)）
#     print(f"Trial {i} eeg shape: {eeg_np.shape}")
#
#     X.append(eeg_np)
#
# X = np.array(X)  # 变成 (trials, channels, timepoints)
# print(f"最终 X 形状: {X.shape}")

#data pre-processing

#visualization
def plot_pre_post_comparison(raw, raw_clean, ch_idx=0):
    import matplotlib.pyplot as plt

    data_raw = raw.get_data()
    data_clean = raw_clean.get_data()
    times = raw.times

    plt.figure(figsize=(15, 4))
    plt.plot(times, data_raw[ch_idx], label='原始', alpha=0.6)
    plt.plot(times, data_clean[ch_idx], label='清洗后', alpha=0.6)
    plt.xlabel('时间 (秒)')
    plt.ylabel('电压 (uV)')
    plt.title(f'通道 {raw.ch_names[ch_idx]}：处理前 vs 处理后')
    plt.legend()
    plt.tight_layout()
    plt.show()


# 假设 raw_data 为你的原始 EEG 信号: shape (4907, 32)
def data_per_process(raw_data, channel_names):
    fs = 500  # sample rate

    # 创建 info 对象（信道信息）
    ch_names = channel_names
    ch_types = ['eeg'] * 32
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    # 转置成 shape (n_channels, n_times)
    raw_data_T = raw_data.T

    # 创建 RawArray 对象
    raw = mne.io.RawArray(raw_data_T, info)

    # 设置标准电极布局（解决 No digitization points）
    raw.set_montage('standard_1020')

    # 1.带通滤波
    raw.filter(0.1, 40., fir_design='firwin', phase='zero',
               l_trans_bandwidth=0.05, h_trans_bandwidth=2.5)

    # 2.common average referencing (CAR)
    raw.set_eeg_reference('average', projection=False)  # 不用投影矩阵，直接应用 CAR

    # 3.ICA
    ica = mne.preprocessing.ICA(n_components=15, method='fastica', random_state=42)
    ica.fit(raw)

    # 可视化用于手动判断眼动伪迹
    ica.plot_components()        # 查看哪些成分可能是伪迹
    ica.plot_sources(raw)        # 查看各独立成分的时间序列

    # 假设你找到第0号和第2号是伪迹（实际需人工判断）
    ica.exclude = [0, 2]

    # 应用 ICA 修正
    raw_clean = ica.apply(raw.copy())

    plot_pre_post_comparison(raw, raw_clean, ch_idx=0)  # 可视化

    return raw_clean


# 提取所有实验（确保是 list of np.ndarray）
eeg_list = [EEG[0, i] for i in range(28)]  # 每个是 (4097, 32)

# 在时间轴（axis=0）上拼接
eeg_concat = np.concatenate(eeg_list, axis=0)  # shape = (4097*28, 32) = (114716, 32)

raw=eeg_concat
raw_clean=data_per_process(eeg_concat, channel_names=eeg_channel_ch_names)  #取一次实验的EEG数据



def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    对信号进行带通滤波，返回滤波后的结果。
    data: shape (T, C)
    """
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')

    # 计算合理的 padlen，不能超过数据长度的一半
    padlen = min(300, data.shape[0] // 2 - 1)

    if padlen <= 0:
        print("[Warning] 信号太短，跳过滤波。")
        return data

    try:
        return sosfiltfilt(sos, data, axis=0, padlen=padlen)
    except Exception as e:
        print(f"[滤波失败] padlen={padlen}, len={len(data)}，错误信息：{e}")
        return data

def downsample_signal(data, original_fs, target_fs):
    """
    使用抗混叠滤波器进行下采样。
    data: shape (T, C)
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    up = 1
    down = int(original_fs / target_fs)

    try:
        return resample_poly(data, up, down, axis=0)
    except Exception as e:
        print(f"[下采样失败] down={down}, len={len(data)}，错误信息：{e}")
        return data

def preprocess_dataset(dataset, original_fs=4000, target_fs=500, lowcut=20, highcut=450):
    processed = []

    for i, sample in enumerate(dataset):
        if not isinstance(sample, np.ndarray):
            print(f"[跳过] 第{i}个样本不是 np.ndarray，类型为 {type(sample)}")
            continue

        try:
            # 滤波
            filtered = bandpass_filter(sample, lowcut, highcut, original_fs)

            # 下采样
            downsampled = downsample_signal(filtered, original_fs, target_fs)

            processed.append(downsampled)
        except Exception as e:
            print(f"[预处理失败] 第{i}个样本，错误信息：{e}")
            continue

    return processed

# # -----------------------
# # 模拟你的数据（28个样本，每个是39256个时间点，5通道）
# data = [EMG[0, i] for i in range(28)]
#
# # 预处理数据
# processed_data = preprocess_dataset(data)
#
# print("原始 shape:", (len(data), data[0].shape))
# for i in range(0,28):
#     print("处理后 shape:", processed_data[i].shape)  # 应为 (28, ~4907, 5)


