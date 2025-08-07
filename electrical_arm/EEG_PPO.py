import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
from model250 import EEGTransformerModel
from torch.utils.data import Dataset, DataLoader
import os

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class EEGDataset(Dataset):
    def __init__(self, eeg):
        self.eeg_data = eeg

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        return x


class PandaEnv(gym.Env):
    def __init__(self, eeg_data, model_path, device):
        super(PandaEnv, self).__init__()

        self.model = mujoco.MjModel.from_xml_path(
            'scene.xml')
        self.data = mujoco.MjData(self.model)
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)\
        # camera location
        self.handle.cam.distance = 3
        self.handle.cam.azimuth = 0
        self.handle.cam.elevation = -30

        # 动作空间，7个关节
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,))
        # 观测空间，包含关节位置和目标位置
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7 + 3,))

        # EEG 模型和数据
        self.device = device
        self.eeg_data = eeg_data
        self.eeg_index = 0
        self.eeg_model = EEGTransformerModel().to(self.device) #实例化模型
        self.eeg_model.load_state_dict(torch.load(model_path, map_location=device)) #加载模型权重
        self.eeg_model.eval() #设置为验证模式

        # 初始目标点
        self.goal = np.zeros(3)
        self.update_goal()
        self.np_random = None

    def update_goal(self):
        if self.eeg_index >= len(self.eeg_data):
            self.eeg_index = 0  # or raise StopIteration if needed

        eeg_input = torch.tensor(self.eeg_data[self.eeg_index], dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.eeg_model(eeg_input)  # shape: [1, 6]

        output = output.squeeze().cpu().numpy()
        finger = output[:3]
        thumb = output[3:]
        midpoint = (finger + thumb) / 2

        # 映射 [-1, 1] → [0.2, 0.6] 正空间（可以根据你的仿真空间调整）
        midpoint = (midpoint + 1) / 2
        self.goal = midpoint * 0.4 + 0.2

        self.eeg_index += 1

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.update_goal()  # 从 EEG 模型生成目标
        print("goal:", self.goal)
        obs = np.concatenate([self.data.qpos[:7], self.goal])
        return obs, {}

    def step(self, action):
        self.data.qpos[:7] = action
        mujoco.mj_step(self.model, self.data)

        achieved_goal = self.data.body(self.end_effector_id).xpos
        reward = -np.linalg.norm(achieved_goal - self.goal)
        reward -= 0.3*self.data.ncon

        terminated = np.linalg.norm(achieved_goal - self.goal) < 0.01
        truncated = False
        info = {'is_success': terminated}

        obs = np.concatenate([self.data.qpos[:7], achieved_goal])

        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.handle.sync()

        self.update_goal()  # decoding EEG get next goal

        return obs, reward, terminated, truncated, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]



# EEG docoding test
def evaluate(model, test_loader, device):
    model.eval()  # 指定模型为验证模式
    out_kin = []

    with torch.no_grad():
        for eeg in test_loader:  # 改变量名
            eeg_batch = eeg.to(device)

            outputs = model(eeg_batch)
            out_kin.append(outputs)

    return out_kin

def EEG_loader():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)

    data_dir=os.path.join(parent_dir, 'dataset')
    model_dir = os.path.join(parent_dir, 'f_model/4_best_model0.86_t250_s50_w200.pth')

    save_path_train_eeg = os.path.join(data_dir, 'train/eeg_train.npy')
    eeg_train = np.load(save_path_train_eeg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EEGTransformerModel()
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.to(device)

    train_dataset = EEGDataset(eeg_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

    vector = evaluate(model, train_loader, device)

    print("验证结果：", vector)



def PPO(eeg_data, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def env_fn():
        return PandaEnv(eeg_data, model_path, device)

    env = make_vec_env(env_fn, n_envs=1)

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        device=device,
        tensorboard_log="./tensorboard/"
    )

    model.learn(total_timesteps=2048*100)
    model.save("panda_ppo_model")



if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)

    data_dir=os.path.join(parent_dir, 'dataset')
    model_dir = os.path.join(parent_dir, 'f_model/4_best_model0.86_t250_s50_w200.pth')

    save_path_train_eeg = os.path.join(data_dir, 'train/eeg_train.npy')

    eeg_train= np.load(save_path_train_eeg)
    PPO(eeg_train, model_dir)

    # eeg_train = np.load(save_path_train_eeg)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # model = EEGTransformerModel()
    # model.load_state_dict(torch.load(model_dir, map_location=device))
    # model.to(device)
    #
    # train_dataset = EEGDataset(eeg_train)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    #
    # vector = evaluate(model, train_loader, device)
    #
    # print("验证结果：", vector)

#你也可以在训练后加载模型，用 EEG 数据继续进行 Fine-Tune（少量更新）
model = PPO.load("panda_ppo_model")
model.set_env(EEGEnv(...))  # EEG 生成目标的环境
model.learn(total_timesteps=2048*10)  # 微调