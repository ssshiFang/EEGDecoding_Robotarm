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
        # print(KIN[0][i].shape)  # motion data

        # Get 3D data of P2 and P3
        # kin_selected_data = KIN[0][i][:, 18:30]  # shape: [T, 6], corresponds to xyz of P2 and P3
        # # print(KIN[0][0][100])  # all 45 channels at frame 100

        # # Extract P2 (channels 1, 5, 9) and P3 (channels 2, 6, 10)
        # kin_p2_p3 = kin_selected_data[:, [1, 5, 9, 2, 6, 10]]  # shape: (T, 6)

        # p2 = kin_p2_p3[:, :3]  # xyz of P2
        # p3 = kin_p2_p3[:, 3:]  # xyz of P3

        # Get 3D coordinates of P2 and P3
        kin_p2_p3 = kin_vector[0][i][:, [19, 23, 27, 20, 24, 28]]  # shape: (T, 6) (px2, py2, pz2, px3, py3, pz3)

        # min-max
        kin_p2_p3_norm = (kin_p2_p3 - kin_p2_p3.min(axis=0)) / (kin_p2_p3.max(axis=0) - kin_p2_p3.min(axis=0) + 1e-8)

        kin_list.append(kin_p2_p3_norm.T)

    return kin_list



def data_preprocess(data):
    my_data = data['ws'] # data saved here
    ws_obj = my_data[0, 0]

    names_obj = ws_obj['names']
    # print(names_obj[0,0].dtype.names) #('eeg', 'kin', 'emg')location
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


def arm_trial(data, have_emg, trial_num):
    my_data = data['ws'] # data saved here
    ws_obj = my_data[0, 0]

    names_obj = ws_obj['names']
    # print(names_obj[0,0].dtype.names) #('eeg', 'kin', 'emg')location
    eeg_channel_names = names_obj['eeg']
    eeg_channel_ch_names = [ch[0] for ch in eeg_channel_names[0][0][0]]

    win_obj=ws_obj['win']

    KIN = win_obj['kin']
    EEG = win_obj['eeg']
    if have_emg:
        EMG = win_obj['emg']
        emg_data = EMG[0, trial_num] # [(T, 5), ]

        # filter
        filtered = bandpass_filter(emg_data, lowcut=20, highcut=450, fs=4000)

        # downsample
        downsampled = downsample_signal(filtered, original_fs=4000, target_fs=500)

        emg_5_channel= downsampled.T

    else:
        emg_5_channel=None

    kin_p2_p3 = KIN[0][trial_num][:, [19, 23, 27, 20, 24, 28]]
    #  min-max
    kin_p2_p3_norm = (kin_p2_p3 - kin_p2_p3.min(axis=0)) / (kin_p2_p3.max(axis=0) - kin_p2_p3.min(axis=0) + 1e-8)
    kin_p2_p3_norm = kin_p2_p3_norm.T

    eeg_32_channel=eeg_process(EEG[0][trial_num], eeg_channel_ch_names)

    # eeg_32_channel = np.concatenate(EEG_processed, axis=1)
    # kin_p2_p3_norm = np.concatenate(KIN_processed, axis=1)

    return eeg_32_channel, kin_p2_p3_norm, emg_5_channel



#emg pre-process
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')

    # padlen no longer than half of length
    padlen = min(300, data.shape[0] // 2 - 1)

    if padlen <= 0:
        print("[Warning] data length not enough")
        return data

    try:
        return sosfiltfilt(sos, data, axis=0, padlen=padlen)
    except Exception as e:
        print(f"[fail] padlen={padlen}, len={len(data)}, {e}")
        return data



def downsample_signal(data, original_fs, target_fs):

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    up = 1
    down = int(original_fs / target_fs)

    try:
        return resample_poly(data, up, down, axis=0)
    except Exception as e:
        print(f"[fail] down={down}, len={len(data)}: {e}")
        return data



def preprocess_dataset(dataset, original_fs=4000, target_fs=500, lowcut=20, highcut=450):
    processed = []

    for i, sample in enumerate(dataset):
        if not isinstance(sample, np.ndarray):
            print(f"{i} sample error, not np.ndarray type:{type(sample)}")
            continue

        try:
            # filter
            filtered = bandpass_filter(sample, lowcut, highcut, original_fs)

            # downsample
            downsampled = downsample_signal(filtered, original_fs, target_fs)

            processed.append(downsampled.T)
        except Exception as e:
            print(f"fail {i}:{e}")
            continue

    return processed



def emg_data_preprocess(data):
    my_data = data['ws']  # data saved here
    ws_obj = my_data[0, 0]

    # obj location
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
    plt.plot(times, data_raw[ch_idx], label='origin', alpha=0.6)
    plt.plot(times, data_clean[ch_idx], label='filtered', alpha=0.6)
    plt.xlabel('s')
    plt.ylabel('uV')
    plt.title(f'pass {raw.ch_names[ch_idx]}after process')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ISA analysis
def eeg_esi_process(raw_data, channel_names):
    fs = 500  # sampling rate

    # Create info object
    ch_names = channel_names
    ch_types = ['eeg'] * 32
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    # Transpose data to shape (n_channels, n_times)
    raw_data_T = raw_data.T

    # Create RawArray object
    raw = mne.io.RawArray(raw_data_T, info)

    # Set standard electrode layout (fixes "No digitization points" issue)
    raw.set_montage('standard_1020')

    # 1. IIR filter
    raw.filter(0.1, 40., fir_design='firwin', phase='zero',
               l_trans_bandwidth=0.05, h_trans_bandwidth=2.5)

    # 2. Common average referencing (CAR)
    raw.set_eeg_reference('average', projection=False)  # apply CAR directly, without projection matrix

    # 3. ICA (Independent Component Analysis)
    ica = mne.preprocessing.ICA(n_components=15, method='fastica', random_state=42)
    ica.fit(raw)

    # Visualize components for manual identification of eye-movement artifacts
    ica.plot_components()        # see which components might be artifacts
    ica.plot_sources(raw)        # see time series of each independent component

    # Suppose components 0 and 2 are artifacts (manual inspection required)
    ica.exclude = [0, 2]

    # Apply ICA to remove selected artifacts
    raw_clean = ica.apply(raw.copy())

    # Visualize before and after ICA for the first channel
    plot_pre_post_comparison(raw, raw_clean, ch_idx=0)

    return raw_clean



# EEG shape (T, 32)
def eeg_process(raw_data, channel_names):
    fs = 500  # sample rate

    # create info object
    ch_names = channel_names
    ch_types = ['eeg'] * 32
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    # shape (n_channels, n_times)
    raw_data_T = raw_data.T

    # create RawArray object
    raw = mne.io.RawArray(raw_data_T, info)

    # normal po location
    raw.set_montage('standard_1020')

    # 1.IIR filiter
    raw.filter(0.1, 40., method='iir')

    # 2.common average referencing (CAR)
    raw.set_eeg_reference('average', projection=False)  # 不用投影矩阵，直接应用 CAR

    data, times = raw.get_data(return_times=True)

    return data


def trials_to_pickle(new_trials, file_path='all_trials.pkl'):
    # adjust id document exist
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            trials = pickle.load(f)
    else:
        trials = []

    # extend trial
    trials.extend(new_trials)

    with open(file_path, 'wb') as f:
        pickle.dump(trials, f)

    print(f"total save {len(trials)} trial")


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


def eeg_arm_pickle(new_trials, file_path='all_trials.pkl'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            trials = pickle.load(f)
    else:
        trials = []

    trials.extend([new_trials]) # add new trial

    with open(file_path, 'wb') as f:
        pickle.dump(trials, f)

    print(f"total save {len(trials)} trial")


def save_eeg_trial(save_path_data, have_emg=False, participant=4, file=1, trial_num=1):
    data_dir = f'D:/MyFolder/Msc_EEG/data{participant}'
    file = f'WS_P{participant}_S{file}.mat'

    file_path = os.path.join(data_dir, file)
    mat = scipy.io.loadmat(file_path)
    eeg, kin, emg = arm_trial(mat, have_emg=have_emg, trial_num=trial_num)

    if have_emg:
        new_trial = {'eeg': eeg, 'emg': emg, 'kin': kin}
    else:
        new_trial = {'eeg': eeg, 'kin': kin}

    eeg_arm_pickle(new_trial, file_path=save_path_data)


def split_trials_fixed(filepath, val_num=30, test_num=30, train_num=234, seed=42):
    with open(filepath, 'rb') as f:
        trials = pickle.load(f)

    # get random
    random.seed(seed)
    random.shuffle(trials)

    # slice
    val_trials = trials[-val_num:]  # last val_num
    test_trials = trials[-(val_num + test_num):-val_num]
    train_trials = trials[:-(val_num + test_num)]

    print(f"Slice outcome  Train: {len(train_trials)}, Val: {len(val_trials)}, Test: {len(test_trials)}")

    return train_trials, val_trials, test_trials


def slice_and_merge_all_trials(trials,
                               save_dir='sliced_all',
                               window_size=250,
                               step_size=50,
                               eeg_name='eeg',
                               kin_name='kin',
                               emg_name='emg',
                               have_emg=False,
                               kin_delay_ms=200,
                               fs=500):  # sample rate

    os.makedirs(save_dir, exist_ok=True)

    all_eeg_slices = []
    all_kin_slices = []
    if have_emg:
        all_emg_slices = []

    kin_delay = int((kin_delay_ms / 1000) * fs)  # delay = 200ms * 500Hz = 100

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

            # check border
            if kin_end > T:
                break

            kin_seg = kin[:, kin_start:kin_end]


            # pedding
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

        print(f" Trial {idx:02d} sliced, total: {int((T - window_size - kin_delay) / step_size + 1)} ")

    all_eeg_array = np.stack(all_eeg_slices)  # shape: (N, 32, window)
    all_kin_array = np.stack(all_kin_slices)  # shape: (N, 6, window)


    # save
    np.save(os.path.join(save_dir, eeg_name), all_eeg_array)
    np.save(os.path.join(save_dir, kin_name), all_kin_array)

    print(f"\n all trial saved:")

    if have_emg:
        all_emg_array = np.stack(all_emg_slices) # shape: (N, 5, window)
        np.save(os.path.join(save_dir, emg_name), all_emg_array)
        print(f"EMG: {all_emg_array.shape}")

    print(f"EEG: {all_eeg_array.shape}, KIN: {all_kin_array.shape}")
    print(f"save path：{save_dir}")


def main():
    # load data
    # data = scipy.io.loadmat('D:/MyFolder/Msc_EEG/data/WS_P5_S1.mat') # dict type

    # get location
    current_dir = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.join(current_dir, 'processed')
    model_dir = os.path.join(current_dir, 'dataset')

    os.makedirs(save_dir, exist_ok=True)

    # save path
    save_path = os.path.join(save_dir, 'subject_all')
    save_path_data = os.path.join(save_dir, 'subject_all/all_trials.pkl')

    save_path_train = os.path.join(model_dir, 'train')
    save_path_test = os.path.join(model_dir, 'test')
    save_path_val = os.path.join(model_dir, 'val')

    have_emg = True # if use multi-modality

    save_all(save_path_data=save_path_data, have_emg=have_emg, participant=1)  # save all data as pkl document

    # get different trials
    train_trials, val_trials, test_trials=split_trials_fixed(save_path_data)

    # slice
    if not(have_emg):
        slice_and_merge_all_trials(train_trials, save_dir=save_path_train, eeg_name='eeg_train.npy', kin_name='kin_train.npy')
        slice_and_merge_all_trials(test_trials, save_dir=save_path_test, eeg_name='eeg_test.npy', kin_name='kin_test.npy')
        slice_and_merge_all_trials(val_trials, save_dir=save_path_val, eeg_name='eeg_val.npy', kin_name='kin_val.npy')
    else:
        slice_and_merge_all_trials(train_trials, save_dir=save_path_train, eeg_name='eeg_train.npy', kin_name='kin_train.npy', emg_name='emg_train.npy', have_emg=have_emg)
        slice_and_merge_all_trials(test_trials, save_dir=save_path_test, eeg_name='eeg_test.npy', kin_name='kin_test.npy', emg_name='emg_test.npy', have_emg=have_emg)
        slice_and_merge_all_trials(val_trials, save_dir=save_path_val, eeg_name='eeg_val.npy', kin_name='kin_val.npy', emg_name='emg_val.npy', have_emg=have_emg)


def check_single_pkl_file(pkl_path, expected_channels=32):
    """
    check pkl data
    """
    if not os.path.exists(pkl_path):
        print(f"do not have {pkl_path}")
        return

    with open(pkl_path, 'rb') as f:
        try:
            trials = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] can not read {pkl_path}: {e}")
            return

        for i, trial in enumerate(trials):
            if isinstance(trial, list):
                trial = trial[0]

            eeg = trial.get('eeg', None)
            if eeg is None:
                print(f"[WARNING] trial {i} not have 'eeg'")
                continue

            if isinstance(eeg, list):
                eeg = eeg[0]

            if hasattr(eeg, 'shape'):
                if eeg.shape[0] != expected_channels:
                    print(f"trial {i} eeg shape: {eeg.shape}, channel: {expected_channels}")
                else:
                    print(f"trial {i}  eeg shape: {eeg.shape}")
            else:
                print(f"trial {i} not have shape")


# eeg data for robot arm
def main2():
    # location
    current_dir = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.join(current_dir, 'electrical_arm')

    os.makedirs(save_dir, exist_ok=True)

    # save path
    save_path = os.path.join(save_dir, 'trial_data')
    save_path_data = os.path.join(save_dir, 'trial_data/arm_trial.pkl')

    have_emg = True

    save_eeg_trial(save_path_data=save_path_data, have_emg=have_emg, participant=9)  # save signal trial for robot arm decoding



if __name__ == "__main__":
    main2()
    # check_single_pkl_file('D:/MyFolder/Msc_EEG/model/Mscproject/EEGtranformer/processed/subject_all/4_all_trials.pkl')