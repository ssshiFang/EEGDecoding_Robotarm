import scipy.io
import numpy as np
import mne
import os
import pickle
import random
from scipy.signal import butter, sosfiltfilt, resample_poly

#eeg kin pre-process
def kin_process(kin_vector):
    kin_list=[]
    for i in range(0,len(kin_vector[0])):
        # print(KIN[0][i].shape) #运动数据
        #获得P2 P3的三维数据
        # kin_selected_data = KIN[0][i][:, 18:30]  # shape: [T, 6]，对应 P2 和 P3 的 xyz
        # # print(KIN[0][0][100])  # 第100帧的所有45通道数据
        #
        # # 提取 P2 (1, 5, 9) 和 P3 (2, 6, 10)
        # kin_p2_p3 = kin_selected_data[:, [1, 5, 9, 2, 6, 10]]  # shape: (T, 6)
        #
        # p2 = kin_p2_p3[:, :3]
        # p3 = kin_p2_p3[:, 3:]

        #获得P2 P3的三维数据
        kin_p2_p3 = kin_vector[0][i][:, [19, 23, 27, 20, 24, 28]]  # shape: (T, 6) (px2, py2, pz2, px3, py3, pz3)

        # 对每一列进行 min-max 归一化
        kin_p2_p3_norm = (kin_p2_p3 - kin_p2_p3.min(axis=0)) / (kin_p2_p3.max(axis=0) - kin_p2_p3.min(axis=0) + 1e-8)

        kin_list.append(kin_p2_p3_norm.T)

    return kin_list



def data_preprocess(data):
    my_data = data['ws'] # 数据文件存储在此
    ws_obj = my_data[0, 0]

    #获得传感器的位置信息
    names_obj = ws_obj['names']
    # print(names_obj[0,0].dtype.names) #('eeg', 'kin', 'emg')分别的位置信息
    eeg_channel_names = names_obj['eeg']
    eeg_channel_ch_names = [ch[0] for ch in eeg_channel_names[0][0][0]]

    win_obj=ws_obj['win']

    KIN = win_obj['kin']
    EEG = win_obj['eeg']
    eeg_32_channel = [] #[()]
    kin_p2_p3_norm = kin_process(KIN)

    for i in range(0,len(EEG[0])):
        eeg_32_channel.append(eeg_process(EEG[0][i], eeg_channel_ch_names))

    # eeg_32_channel = np.concatenate(EEG_processed, axis=1)
    # kin_p2_p3_norm = np.concatenate(KIN_processed, axis=1)

    return eeg_32_channel, kin_p2_p3_norm



#emg pre-process
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

            processed.append(downsampled.T)
        except Exception as e:
            print(f"[预处理失败] 第{i}个样本，错误信息：{e}")
            continue

    return processed



def emg_data_preprocess(data):
    my_data = data['ws']  # 数据文件存储在此
    ws_obj = my_data[0, 0]

    # 获得传感器的位置信息
    names_obj = ws_obj['names']
    win_obj=ws_obj['win']
    EMG = win_obj['emg']

    emg_data = [EMG[0, i] for i in range(28)]

    processed_data = preprocess_dataset(emg_data) #[(T, 5), ]
    # 5-anterior deltoid, brachioradial, flexor digitorum, common extensor digitorum, and the first dorsal interosseus muscles

    return processed_data



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


# ISA分析
def eeg_esi_process(raw_data, channel_names):
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


# 假设 raw_data 为你的原始 EEG 信号: shape (4907, 32)
def eeg_process(raw_data, channel_names):
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

    # 1.IIR滤波
    raw.filter(0.1, 40., method='iir')

    # 2.common average referencing (CAR)
    raw.set_eeg_reference('average', projection=False)  # 不用投影矩阵，直接应用 CAR

    data, times = raw.get_data(return_times=True)

    return data


# def slice_save(eeg_trials, kin_trials, window_size=1000, step_size=250, save_path='./sliced_data'):
#     all_eeg_slices = []
#     all_kin_slices = []
#
#     os.makedirs(save_path, exist_ok=True)
#
#     for idx, (eeg, kin) in enumerate(zip(eeg_trials, kin_trials)):
#         assert eeg.shape[1] == kin.shape[1], f"第 {idx} 个 trial 时间长度不一致"
#         n_times = eeg.shape[1]
#
#         start = 0
#         while start < n_times:
#             end = start + window_size
#
#             # EEG slice
#             eeg_slice = eeg[:, start:end]
#             if eeg_slice.shape[1] < window_size:
#                 pad = window_size - eeg_slice.shape[1]
#                 eeg_slice = np.pad(eeg_slice, ((0, 0), (0, pad)), mode='constant')
#
#             # KIN slice
#             kin_slice = kin[:, start:end]
#             if kin_slice.shape[1] < window_size:
#                 pad = window_size - kin_slice.shape[1]
#                 kin_slice = np.pad(kin_slice, ((0, 0), (0, pad)), mode='constant')
#
#             all_eeg_slices.append(eeg_slice)
#             all_kin_slices.append(kin_slice)
#
#             # 若遇到补零，停止滑动
#             if end >= n_times:
#                 break
#
#             start += step_size
#
#     # 合并为 numpy 数组
#     eeg_array = np.stack(all_eeg_slices)  # shape: (num_windows, 32, 1000)
#     kin_array = np.stack(all_kin_slices)  # shape: (num_windows, 6, 1000)
#
#     # 保存
#     np.save(os.path.join(save_path, 'sliced_eeg.npy'), eeg_array)
#     np.save(os.path.join(save_path, 'sliced_kin.npy'), kin_array)
#
#     print(f"保存完成：{eeg_array.shape[0]} 段")
#     print(f"EEG shape: {eeg_array.shape}, Kin shape: {kin_array.shape}")


#保存为csv文件
# import numpy as np
# import pandas as pd
# import os
#
# def save_trials_as_csv(eeg_trials, kin_trials, save_dir='csv_trials'):
#     os.makedirs(save_dir, exist_ok=True)
#
#     for i, (eeg, kin) in enumerate(zip(eeg_trials, kin_trials)):
#         assert eeg.shape[1] == kin.shape[1], f"第 {i} 个 trial 时间长度不一致"
#
#         # 转置：每一行为一个时间点
#         eeg_T = eeg.T  # (T, 32)
#         kin_T = kin.T  # (T, 6)
#         combined = np.concatenate([eeg_T, kin_T], axis=1)  # (T, 38)
#
#         # 构造列名
#         eeg_cols = [f'EEG_ch{j+1}' for j in range(eeg.shape[0])]
#         kin_cols = ['P2_x', 'P2_y', 'P2_z', 'P3_x', 'P3_y', 'P3_z']
#         col_names = eeg_cols + kin_cols
#
#         df = pd.DataFrame(combined, columns=col_names)
#         csv_path = os.path.join(save_dir, f'trial_{i:02d}.csv')
#         df.to_csv(csv_path, index=False)
#
#     print(f"共保存 {len(eeg_trials)} 个 trial 到文件夹：{save_dir}")



def trials_to_pickle(new_trials, file_path='all_trials.pkl'):
    # 1. 先判断文件是否存在
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            trials = pickle.load(f)
    else:
        trials = []

    # 2. 添加新 trial
    trials.extend(new_trials)

    # 3. 覆写保存
    with open(file_path, 'wb') as f:
        pickle.dump(trials, f)

    print(f"当前文件中共保存 {len(trials)} 个 trial")


def save_all(save_path_data, have_emg=False, participant=1):
    data_dir = f'D:/MyFolder/Msc_EEG/data{participant}'
    file_list = sorted([f for f in os.listdir(data_dir) if f.startswith(f'WS_P{participant}_S') and f.endswith('.mat')])

    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        mat = scipy.io.loadmat(file_path)
        eeg,kin=data_preprocess(mat)

        #have emg add this code
        emg=emg_data_preprocess(mat)

        if have_emg:
            new_trials = [{'eeg': eeg,'emg': emg, 'kin': kin} for eeg, emg, kin in zip(eeg, emg, kin)]
        else:
            new_trials = [{'eeg': eeg, 'kin': kin} for eeg, kin in zip(eeg, kin)]


        trials_to_pickle(new_trials, file_path=save_path_data)


def split_trials_fixed(filepath, val_num=60, test_num=60, train_num=300, seed=42):
    # 1. 加载数据
    with open(filepath, 'rb') as f:
        trials = pickle.load(f)

    # 2. 打乱
    random.seed(seed)
    random.shuffle(trials)

    # 3. 划分
    val_trials = trials[:val_num]
    test_trials = trials[val_num:val_num + test_num]
    train_trials = trials[val_num + test_num:val_num + test_num + train_num]

    print(f"划分结果：Train: {len(train_trials)}, Val: {len(val_trials)}, Test: {len(test_trials)}")

    return train_trials, val_trials, test_trials


# 按trials保存
# def slice_and_save_per_trial(trials, save_dir='sliced_trials', window_size=1000, step_size=250):
#     os.makedirs(save_dir, exist_ok=True)
#
#     for idx, trial in enumerate(trials):
#         eeg = trial['eeg']  # shape: (32, T)
#         kin = trial['kin']  # shape: (6, T)
#         T = eeg.shape[1]
#
#         eeg_slices = []
#         kin_slices = []
#
#         for start in range(0, T, step_size):
#             end = start + window_size
#
#             eeg_seg = eeg[:, start:end]
#             kin_seg = kin[:, start:end]
#
#             if eeg_seg.shape[1] < window_size:
#                 pad = window_size - eeg_seg.shape[1]
#                 eeg_seg = np.pad(eeg_seg, ((0, 0), (0, pad)))
#                 kin_seg = np.pad(kin_seg, ((0, 0), (0, pad)))
#
#             eeg_slices.append(eeg_seg)
#             kin_slices.append(kin_seg)
#
#             if end >= T:
#                 break  # 不再往后滑动
#
#         # 保存本 trial 的所有切片为 .npy
#         eeg_array = np.stack(eeg_slices)  # shape: (n_windows, 32, 1000)
#         kin_array = np.stack(kin_slices)  # shape: (n_windows, 6, 1000)
#
#         np.save(os.path.join(save_dir, f'trial_{idx:02d}_eeg.npy'), eeg_array)
#         np.save(os.path.join(save_dir, f'trial_{idx:02d}_kin.npy'), kin_array)
#
#         print(f"Trial {idx:02d} 保存：{eeg_array.shape[0]} 段")
#
#     print(f"🎉 所有 trial 保存完毕，路径：{save_dir}")


def slice_and_merge_all_trials(trials,
                               save_dir='sliced_all',
                               window_size=250,
                               step_size=50,
                               eeg_name='eeg',
                               kin_name='kin',
                               emg_name='emg',
                               have_emg=False,
                               kin_delay_ms=200,
                               fs=500):  # 采样率

    os.makedirs(save_dir, exist_ok=True)

    all_eeg_slices = []
    all_kin_slices = []
    if have_emg:
        all_emg_slices = []

    kin_delay = int((kin_delay_ms / 1000) * fs)  # 延迟点数 = 200ms * 500Hz = 100点

    for idx, trial in enumerate(trials):
        eeg = trial['eeg']  # shape: (32, T)
        kin = trial['kin']  # shape: (6, T)
        if have_emg:
            emg = trial['emg']
        T = eeg.shape[1]

        for start in range(0, T, step_size):
            end = start + window_size

            # EEG segment
            eeg_seg = eeg[:, start:end]

            # KIN segment with delay
            kin_start = start + kin_delay
            kin_end = end + kin_delay

            # 边界检查（避免越界）
            if kin_end > T:
                break  # 超出边界，放弃这个窗口（也可考虑 pad）

            kin_seg = kin[:, kin_start:kin_end]


            # 补零（仅针对 EEG，不处理越界 KIN）
            if eeg_seg.shape[1] < window_size:
                pad = window_size - eeg_seg.shape[1]
                eeg_seg = np.pad(eeg_seg, ((0, 0), (0, pad)))

            all_eeg_slices.append(eeg_seg)
            all_kin_slices.append(kin_seg)

            # EMG segment
            if have_emg:
                emg_start = start + kin_delay
                emg_end = end + kin_delay
                emg_seg = emg[:, emg_start:emg_end]

                all_emg_slices.append(emg_seg)

        print(f" Trial {idx:02d} 切片完成，共 {int((T - window_size - kin_delay) / step_size + 1)} 段")

    all_eeg_array = np.stack(all_eeg_slices)  # shape: (N, 32, window)
    all_kin_array = np.stack(all_kin_slices)  # shape: (N, 6, window)


    # 保存
    np.save(os.path.join(save_dir, eeg_name), all_eeg_array)
    np.save(os.path.join(save_dir, kin_name), all_kin_array)

    print(f"\n 所有 trial 已拼接并保存：")

    if have_emg:
        all_emg_array = np.stack(all_emg_slices) # shape: (N, 5, window)
        np.save(os.path.join(save_dir, emg_name), all_emg_array)
        print(f"EMG: {all_emg_array.shape}")

    print(f"EEG: {all_eeg_array.shape}, KIN: {all_kin_array.shape}")
    print(f"保存路径：{save_dir}")


def main():
    # 加载数据
    # data = scipy.io.loadmat('D:/MyFolder/Msc_EEG/data/WS_P5_S1.mat') # dict type

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 拼接你想保存的子文件夹路径（如 'processed'）
    save_dir = os.path.join(current_dir, 'processed')
    model_dir = os.path.join(current_dir, 'dataset')

    # 创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 数据的加载保存路径
    save_path = os.path.join(save_dir, 'subject_all')
    save_path_data = os.path.join(save_dir, 'subject_all/all_trials.pkl')

    save_path_train = os.path.join(model_dir, 'train')
    save_path_test = os.path.join(model_dir, 'test')
    save_path_val = os.path.join(model_dir, 'val')

    have_emg=True

    # save_all(save_path_data=save_path_data, have_emg=have_emg, participant=9)  # 将所有实验保存为pkl文件

    train_trials, val_trials, test_trials=split_trials_fixed(save_path_data)

    #对这三个做切分
    if not(have_emg):
        slice_and_merge_all_trials(train_trials, save_dir=save_path_train, eeg_name='eeg_train.npy', kin_name='kin_train.npy')
        slice_and_merge_all_trials(test_trials, save_dir=save_path_test, eeg_name='eeg_test.npy', kin_name='kin_test.npy')
        slice_and_merge_all_trials(val_trials, save_dir=save_path_val, eeg_name='eeg_val.npy', kin_name='kin_val.npy')
    else:
        slice_and_merge_all_trials(train_trials, save_dir=save_path_train, eeg_name='eeg_train.npy', kin_name='kin_train.npy', emg_name='emg_train.npy', have_emg=have_emg)
        slice_and_merge_all_trials(test_trials, save_dir=save_path_test, eeg_name='eeg_test.npy', kin_name='kin_test.npy', emg_name='emg_test.npy', have_emg=have_emg)
        slice_and_merge_all_trials(val_trials, save_dir=save_path_val, eeg_name='eeg_val.npy', kin_name='kin_val.npy', emg_name='emg_val.npy', have_emg=have_emg)



    #文件长度记录
    #5_7_9 882 trial

    # 切分记录，(切片长度，步长)
    # (1000,250)
    # (500, 200)
    # (250, 50)

    #查看整理好的数据
    # with open(save_path_data, 'rb') as f:
    #     trials = pickle.load(f)
    #     print(f"共载入 {len(trials)} 个 trial")
    #
    #     # 查看第一个 trial 的 shape
    #     print("EEG shape:", trials[0]['eeg'].shape)
    #     print("KIN shape:", trials[0]['kin'].shape)

    # slice_save(eeg,kin,save_path=save_path)


if __name__ == "__main__":
    main()