import pickle
import numpy as np
import mne


def ESI_trial(trial, sfreq=500):
    eeg = trial['eeg']
    info = mne.create_info([f'EEG {i}' for i in range(eeg.shape[0])], sfreq, ['eeg'] * eeg.shape[0])
    raw = mne.io.RawArray(eeg, info)
    raw.set_eeg_reference('average')
    raw.filter(0.1, 40)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    subject = 'fsaverage'
    subjects_dir = mne.datasets.fetch_fsaverage(verbose=True)
    src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir)
    model = mne.make_bem_model(subject=subject, ico=4, conductivity=[0.3], subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    trans = 'fsaverage'

    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True)

    events = np.array([[0, 0, 1]])
    event_id = dict(grasp=1)
    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=eeg.shape[1]/sfreq - 1/sfreq, preload=True)
    evoked = epochs.average()
    noise_cov = mne.compute_covariance(epochs, tmax=0.0)
    inv_op = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)
    stc = mne.minimum_norm.apply_inverse(evoked, inv_op, lambda2=1. / 9., method='sLORETA')
    return stc.data  # shape: (n_sources, n_times)



def ESI_structure():
    # 1.load data from .pkl
    with open('path_to_trial.pkl', 'rb') as f:
        trials = pickle.load(f)

    trial = trials[0]  # 任选一个 trial 测试
    eeg_data = trial['eeg']  # shape: (n_channels, n_times)

    # 2.build MNE RawArray object
    n_channels = eeg_data.shape[0]
    sfreq = 500  # 采样率，按你的数据设定
    ch_names = [f'EEG {i}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)

    # 可选：去均值、带通滤波等
    raw.set_eeg_reference('average')
    raw.filter(0.1, 40)

    # 3.head model
    subjects_dir = mne.datasets.fetch_fsaverage(verbose=True)
    subject = 'fsaverage'

    src = mne.setup_source_space(subject, spacing='oct6', add_dist=False, subjects_dir=subjects_dir)

    model = mne.make_bem_model(subject=subject, ico=4, conductivity=[0.3], subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)

    montage = mne.channels.make_standard_montage('standard_1020')  # 或者根据你电极布局自定义
    raw.set_montage(montage)

    trans = 'fsaverage'  # 默认头模转换
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True)

    # 4.create events and create epoch
    # 创建一个 epoch 时间窗口（例如 -0.2s 到 0.5s）
    events = np.array([[0, 0, 1]])  # 一个事件，index 0
    event_id = dict(grasp=1)

    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=eeg_data.shape[1] / sfreq - 1 / sfreq,
                        baseline=None, preload=True)

    evoked = epochs.average()

    # 5.compute cov inverse get esi
    noise_cov = mne.compute_covariance(epochs, tmax=0.0)

    inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)

    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator,
                                         lambda2=1.0 / 9.0, method='sLORETA')

    # 得到源域特征：stc.data.shape = (n_sources, n_times)




