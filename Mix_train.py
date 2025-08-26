import torch
import torch.nn as nn
import torch.optim as optim #优化器
from torch.utils.data import Dataset, DataLoader #数据加载
from tqdm import tqdm #训练进度条
import os
from Mix_control_model250 import EEGTransformerModel
import numpy as np


def pearson_corrcoef(x, y):
    """
    pcc compute
    input x, y - shape tensor 1D\2D
    output PCC（float）
    """
    x = x.float()
    y = y.float()

    vx = x - x.mean()
    vy = y - y.mean()

    numerator = (vx * vy).sum()
    denominator = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8

    return (numerator / denominator).item()


def rmse(x, y):
    """
    rmse compute
    input x, y - shape tensor 1D\2D
    output rmse（float）
    """
    x = x.float()
    y = y.float()
    return torch.sqrt(torch.mean((x - y) ** 2)).item()



def pearson_seperate(x, y): # [px2, py2, pz2, px3, py3, pz3]
    """
    pcc compute:
    x, y shape: (B, 6)
    """
    px2_pre = x[:, 0]
    py2_pre = x[:, 1]
    pz2_pre = x[:, 2]
    px3_pre = x[:, 3]
    py3_pre = x[:, 4]
    pz3_pre = x[:, 5]

    px2 = y[:, 0]
    py2 = y[:, 1]
    pz2 = y[:, 2]
    px3 = y[:, 3]
    py3 = y[:, 4]
    pz3 = y[:, 5]

    pcc = pearson_corrcoef(x, y)  # [px2, py2, pz2, px3, py3, pz3]

    # 假设 shape 是 (batch_size, 2)
    pcc_px2 = pearson_corrcoef(px2_pre, px2)
    pcc_px3 = pearson_corrcoef(px3_pre, px3)
    pcc_x = (pcc_px2 + pcc_px3) / 2

    pcc_py2 = pearson_corrcoef(py2_pre, py2)
    pcc_py3 = pearson_corrcoef(py3_pre, py3)
    pcc_y = (pcc_py2 + pcc_py3) / 2

    pcc_pz2 = pearson_corrcoef(pz2_pre, pz2)
    pcc_pz3 = pearson_corrcoef(pz3_pre, pz3)
    pcc_z = (pcc_pz2 + pcc_pz3) / 2

    rmse_all = rmse(x, y)

    # rmse_px2 = rmse(px2_pre, px2)
    # rmse_px3 = rmse(px3_pre, px3)
    # rmse_x = (rmse_px2 + rmse_px3) / 2
    #
    # rmse_py2 = rmse(py2_pre, py2)
    # rmse_py3 = rmse(py3_pre, py3)
    # rmse_y = (rmse_py2 + rmse_py3) / 2
    #
    # rmse_pz2 = rmse(pz2_pre, pz2)
    # rmse_pz3 = rmse(pz3_pre, pz3)
    # rmse_z = (rmse_pz2 + rmse_pz3) / 2

    result = {
        "pcc": pcc,
        "pcc_x": pcc_x,
        "pcc_y": pcc_y,
        "pcc_z": pcc_z,
        "rmse": rmse_all
        # "rmse_x": rmse_x,
        # "rmse_y": rmse_y,
        # "rmse_z": rmse_z
    }

    return result



class EEGDataset(Dataset):
    def __init__(self, eeg, emg, kin):
        """
        eeg_data: shape [N_samples, N_channels, N_times]
        kin_data: shape [N_samples]
        """
        assert eeg.shape[0] == kin.shape[0], "EEG 和 KIN 数量不一致"
        assert emg.shape[0] == kin.shape[0], "EMG 和 KIN 数量不一致"
        self.eeg_data = eeg
        self.emg_data = emg
        self.kin_data = kin

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        y = torch.tensor(self.emg_data[idx], dtype=torch.float32)
        z = torch.tensor(self.kin_data[idx], dtype=torch.float32)
        
        return x, y, z



def train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs):
    best_pcc = -float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for eeg, emg, kin in tqdm(train_loader, desc=f'epoch:{epoch+1}/{num_epochs}',unit='batch'):
            eeg_batch, emg_batch, kin_batch = eeg.to(device), emg.to(device), kin.to(device) #将数据传到设备上 (B, 32, T), (B, 6, T)

            optimizer.zero_grad()
            outputs = model(eeg_batch, emg_batch)  # 前向传播获得 (B, 6)
            kin_last = kin_batch[:, :, -1]  # 当前段内采样的最后一个坐标点
            loss = criterion(outputs, kin_last) #loss compute
            loss.backward() #反向传播
            optimizer.step() #更新参数

            running_loss += loss.item() * eeg_batch.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)  # 每轮的损失
        val_pcc = evaluate(model, test_loader, device)
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"Train Loss: {epoch_loss:.4f} | "
            f"PCC - PCC: {val_pcc['pcc']:.4f}, X: {val_pcc['pcc_x']:.4f}, Y: {val_pcc['pcc_y']:.4f}, Z: {val_pcc['pcc_z']:.4f}, RMSE: {val_pcc['rmse']:.4f}"
        )

        if val_pcc['pcc'] > best_pcc:
            best_pcc = val_pcc['pcc']
            save_model(model, f'f_model/best_model{best_pcc:.2f}.pth')  # 保存最佳模型



def evaluate(model, test_loader, device):
    model.eval()  # 指定模型为验证模式
    out_kin = []
    kin_list = []  # 改名，避免和循环变量冲突

    with torch.no_grad():
        for eeg, emg, kin in test_loader:  # 改变量名
            eeg_batch, emg_batch, kin_batch = eeg.to(device), emg.to(device), kin.to(device)

            outputs = model(eeg_batch, emg_batch)
            label_last = kin_batch[:, :, -1]

            out_kin.append(outputs)
            kin_list.append(label_last) #[(B.6),(B,6), ]

    preds = torch.cat(out_kin, dim=0)   # N 是整个 test set 的样本数
    kin_tensor = torch.cat(kin_list, dim=0) # N 是相同数量的真实标签 (128, 6)

    result = pearson_seperate(preds, kin_tensor)  # [px2, py2, pz2, px3, py3, pz3]

    return result


# 保存模型
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)



def main():
    num_epochs = 25
    save_path = 'D:/MyFolder/Msc_EEG/model/Mscproject/EEGtranformer/model.pth'
    learning_rate = 0.001
    num_output = 6

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGTransformerModel(output_dim=num_output).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'dataset')

    save_path_train_eeg = os.path.join(model_dir, 'train/eeg_train.npy')
    save_path_train_emg = os.path.join(model_dir, 'train/emg_train.npy')
    save_path_train_kin = os.path.join(model_dir, 'train/kin_train.npy')
    save_path_test_eeg = os.path.join(model_dir, 'test/eeg_test.npy')
    save_path_test_emg = os.path.join(model_dir, 'test/emg_test.npy')
    save_path_test_kin = os.path.join(model_dir, 'test/kin_test.npy')


    # save_path_val_eeg = os.path.join(model_dir, 'val/eeg.val')
    # save_path_val_kin = os.path.join(model_dir, 'val/kin.val')

    # 加载数据
    eeg_train = np.load(save_path_train_eeg)
    emg_train = np.load(save_path_train_emg)
    kin_train = np.load(save_path_train_kin)
    eeg_test = np.load(save_path_test_eeg)
    emg_test = np.load(save_path_test_emg)
    kin_test = np.load(save_path_test_kin)

    train_dataset = EEGDataset(eeg_train, emg_train, kin_train)
    test_dataset = EEGDataset(eeg_test, emg_test, kin_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 开始训练
    train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs)

    # 测试评估
    val_pcc = evaluate(model, test_loader, device)
    print( f"PCC - PCC: {val_pcc['pcc']:.4f}, X: {val_pcc['pcc_x']:.4f}, Y: {val_pcc['pcc_y']:.4f}, Z: {val_pcc['pcc_z']:.4f}, RMSE: {val_pcc['rmse']:.4f}")

    save_model(model, save_path)

def test():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir=os.path.join(current_dir, 'dataset')
    model_dir = os.path.join(current_dir, 'f_model/5_EMG_best_model0.94_t250_s50_w200.pth')

    save_path_val_eeg = os.path.join(data_dir, 'val/eeg_val.npy')
    save_path_val_emg = os.path.join(data_dir, 'val/emg_val.npy')
    save_path_val_kin = os.path.join(data_dir, 'val/kin_val.npy')

    eeg_val = np.load(save_path_val_eeg)
    emg_val = np.load(save_path_val_emg)
    kin_val = np.load(save_path_val_kin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EEGTransformerModel()
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.to(device)

    val_dataset = EEGDataset(eeg_val, emg_val, kin_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    result = evaluate(model, val_loader, device)

    print("验证结果（Pearson correlation）：", result)




if __name__ == "__main__":
    main()

    # #pcc 测试
    # x = torch.randn(128, 6)
    # y = torch.randn(128, 6)
    # result=pearson_seperate(x,y)
    #
    # print(result)

    # val测试
    # test()

