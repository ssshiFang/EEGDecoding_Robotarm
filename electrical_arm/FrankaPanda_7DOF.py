import mujoco
import mujoco.viewer
import numpy
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import torch.nn as nn
import time
import torch
from Mix_model250_256 import EEGTransformerModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import pickle
import os


def align(eeg, emg, delay=200):
    assert eeg.shape[1] == emg.shape[1], "EEG 和 EMG 时间长度必须一致"
    time_steps = eeg.shape[1]

    if delay >= time_steps:
        raise ValueError("延迟值不能大于或等于信号长度")

    new_len = time_steps - delay
    eeg_aligned = eeg[:, :new_len]
    emg_aligned = emg[:, delay:]

    return eeg_aligned, emg_aligned


def trial_data(have_emg= False, kin_output=False):
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)

    # 拼接你想保存的子文件夹路径（如 'processed'）
    save_dir = os.path.join(parent_dir, 'electrical_arm')

    # 数据的加载保存路径
    eeg_path = os.path.join(save_dir, 'trial_data/arm_trial.pkl')

    with open(eeg_path, 'rb') as f:
        trial = pickle.load(f)


    if have_emg:
        eeg_trial = trial[0]['eeg']  # shape: (32, T)
        emg_trial = trial[0]['emg']  # shape: (N_emg, T)
        eeg, emg = align(eeg_trial, emg_trial)
        return eeg, emg

    if kin_output:
        kin_trial = trial[0]['kin']  # shape: (6, T)
        return kin_trial

    else:
        eeg_trial = trial[0]['eeg']  # shape: (32, T)
        return eeg_trial





# EEG docoding test
def EEG_Decoder(eeg_data, emg_data=None, segment_length=250, have_emg=False):
    """
    加载 EEG 模型，对给定 EEG 数据进行分段解码，输出目标点序列
    :eeg_data: numpy array, shape=(32, T) if have_emg= True emg (5, T)
    :param segment_length: 每段的时间步数（默认250）
    :return: numpy array, shape=(N, 6) — 每段一个目标点
    """

    # 获取当前路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # 模型路径
    model_path = os.path.join(parent_dir, 'f_model/5_EMG_best_model0.94_t250_s50_w200.pth')

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = EEGTransformerModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # cut
    C, T = eeg_data.shape
    segments = []

    for i in range(T // segment_length):
        chunk = eeg_data[:, i * segment_length:(i + 1) * segment_length]
        segments.append(chunk)

    if T % segment_length != 0:
        last_chunk = eeg_data[:, -segment_length:]  # 最后一段补足250
        segments.append(last_chunk)

    # 转换为 batch tensor
    eeg_batch = torch.tensor(np.stack(segments), dtype=torch.float32).to(device)  # (N, 32, 250)

    if have_emg:
        C, T = emg_data.shape
        seg = []

        for i in range(T // segment_length):
            cut = emg_data[:, i * segment_length:(i + 1) * segment_length]
            seg.append(cut)

        if T % segment_length != 0:
            last_cut = emg_data[:, -segment_length:]  # 最后一段补足250
            seg.append(last_cut)

        # 转换为 batch tensor
        emg_batch = torch.tensor(np.stack(seg), dtype=torch.float32).to(device)  # (N, 32, 250)

        with torch.no_grad():
            outputs = model(eeg_batch, emg_batch)  # (N, 6)
            positions = outputs.cpu().numpy()

        return positions


    # 分阶段推理

    with torch.no_grad():
        outputs = model(eeg_batch)  # (N, 6)
        positions = outputs.cpu().numpy()

    return positions


# 根据名称 'hand' 查找末端执行器的 body ID
def find_end_site(model):
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    if end_effector_id == -1:
        # 如果未找到指定名称的末端执行器，打印警告信息并终止 GLFW
        print("Warning: Could not find the end effector with the given name.")

    return end_effector_id


def limit_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

# 映射归一化坐标到机械臂工作空间（这里简单映射，按需调整）
def map_workspace(coords):
    workspace_min = np.array([0.3, 0.03, 0.13])
    workspace_max = np.array([0.8,  0.7, 0.55])
    return workspace_min + coords * (workspace_max - workspace_min)

#简单的数值逆运动学（基于雅可比矩阵的伪逆法）
def ik_jacobian(target_pos,
                target_quat,
                site_id,
                model,
                data,
                max_iters=100, # repeat time
                tol=1e-3, # tolerance
                alpha=0.1): # step
    q = data.qpos[:].copy() # site location

    # 获取目标朝向（例如初始朝向）
    mujoco.mj_forward(model, data)
    res = np.zeros(3)

    for _ in range(max_iters):
        data.qpos[:7] = q[:7]
        mujoco.mj_forward(model, data) # depend on pos compute all location
        ee_pos = data.site("ee_site").xpos  # site location
        ee_xmat = data.site(site_id).xmat.reshape(3, 3) # where site point to
        ee_quat_xyzw = R.from_matrix(ee_xmat).as_quat()
        ee_quat = np.roll(ee_quat_xyzw, 1)

        # location loss
        pos_err = target_pos - ee_pos

        # condition loss
        # 使用四元数差异来近似角度误差
        mujoco.mju_subQuat(res, target_quat, ee_quat)
        quat_err = res  # 返回的是 axis-angle 差异

        # 拼接误差向量 [位置误差 + 姿态误差]
        error = np.concatenate([pos_err, quat_err])

        print(f" Error Norm: {np.linalg.norm(error):.4f}")
        print(f"EE Pos: {ee_pos}")

        if np.linalg.norm(error) < tol: # distance of two points
            break

        # 雅可比矩阵
        jacp = np.zeros((3, model.nv)) # IK运动控制 location
        jacr = np.zeros((3, model.nv)) # 姿态控制
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        # 拼接为 6xN 的雅可比
        jac_full = np.vstack([jacp, jacr])[:, :model.nv]

        # # 冻结第7个关节（索引为5），将它列设为0
        # jac_full[:, 6] = 0

        # IK 更新步长
        dq = alpha * np.linalg.pinv(jac_full) @ error
        q[:model.nv] += dq

    return q


def ik_jacobian_position(target_pos,
                         site_id,
                         model,
                         data,
                         max_iters=100,
                         tol=1e-3,
                         alpha=0.1):
    q = data.qpos[:].copy()

    mujoco.mj_forward(model, data)

    for _ in range(max_iters):
        data.qpos[:7] = q[:7]
        mujoco.mj_forward(model, data)

        ee_pos = data.site(site_id).xpos  # 当前末端执行器的位置
        pos_err = target_pos - ee_pos     # 位置误差

        print(f"Error Norm: {np.linalg.norm(pos_err):.4f}")
        print(f"EE Pos: {ee_pos}")

        if np.linalg.norm(pos_err) < tol:
            break

        # 只计算位置的雅可比
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)

        # 使用伪逆计算关节更新
        dq = alpha * np.linalg.pinv(jacp[:, :model.nv]) @ pos_err
        q[:model.nv] += dq

    return q


def initial_location(model,
                     data,
                     hold_time=3.0,
                     steps=500):
    start_q=numpy.zeros(7)
    initial_q=numpy.array([0,0.1,0,-3,2.9,1.65,-2.4]) # initial [0,0.335,0,-3,2.9,1.4,-2.4,0.02,0.02]
    # initial_q = numpy.array([-0.78, 0.1, 0, -3, 2.9, 1.65, -2.4])
    # 活动范围 (x,y 0.3 -0.78 平均 0.24 - 0.55 z 0.12 - 0.65)
    # xy平均-0.782
    # 2号移动范围 -0.4 - 1.55
    # 3号移动范围 -1.8 - 1.5
    # 5号移动范围 2 - 2.9
    # end site location [0.31431234 0.03086261 0.12288893]

    initial_q = initial_q.copy()
    # 设置关节目标位置
    data.qpos[:7] = initial_q

    mujoco.mj_forward(model, data)

    site_id = model.site("ee_site").id
    initial_xmat = data.site(site_id).xmat.reshape(3, 3)
    target_quat = R.from_matrix(initial_xmat).as_quat()

    trajectory = np.linspace(start_q, initial_q[:7], steps)  # zero -> initial
    hold_segment = duration(initial_q, hold_time, model)

    # 拼接完整轨迹
    initial_trajectory = np.vstack((trajectory, hold_segment))  # shape: (N, 7)

    # 打印相关位置
    # end_effector_id = find_end_site()
    # end_effector_pos = data.body(end_effector_id).xpos
    # print("Actuators in model:", model.nu)
    # print(end_effector_pos)
    # print("nq =", model.nq)
    # print("qpos0 =", model.qpos0)
    # print("initial_q = ", data.qpos[:])
    # for i in range(model.njnt):
    #     print(f"Joint {i}: name = {model.joint(i).name}, type = {model.joint(i).type}, addr = {model.jnt_qposadr[i]}")

    # mujoco.mj_step(model, data)是执行一次完整的仿真，mujoco.mj_forward(model, data)只是一次派生求解

    return initial_q, target_quat, initial_trajectory


# # 轨迹生成，笛卡尔空间插值 + 逆运动学（IK）
# def linear_interp(start, end, num_steps):
#     return np.linspace(start, end, num_steps)
#
# # 1. 获取当前位置
# start_pos = sim.data.get_site_xpos("ee_site")  # shape: (3,)
# goal_pos = np.array([0.5, 0.0, 0.3])            # 示例目标位置
#
# # 2. 插值轨迹
# num_steps = 50
# traj = linear_interp(start_pos, goal_pos, num_steps)
#
# joint_trajectory = []
#
# # 3. 对每个插值点做 IK
# for pos in traj:
#     # 这里你需要一个 IK 解算器，如 ik_solver.solve(pos)
#     q = inverse_kinematics(pos)  # shape: (n_joints,)
#     joint_trajectory.append(q)
# 插值进行运动轨迹生成
def generate_joint_trajectory(start, goal, steps):
    return np.linspace(start, goal, steps)


# 追加“保持”部分
def duration(target_q, time_len, model):
    hold_duration = 2.0  # 停留 1 秒
    sim_freq = int(1 / model.opt.timestep)  # 通常是 500~1000 Hz，取决于模型
    hold_steps = int(time_len * sim_freq / 20)  # 降低到 viewer 的视觉频率
    hold_segment = np.tile(target_q, (hold_steps, 1))  # 重复目标姿态

    return hold_segment


def initial(model,data,
            hold_time=3.0,
            steps=500):
    start_q=numpy.zeros(7)
    initial_q=numpy.array([-0.78,0.15,0,-3,2.9,1.65,-2.4]) # initial [0,0.335,0,-3,2.9,1.4,-2.4]

    trajectory = np.linspace(start_q, initial_q, steps) # zero -> initial
    hold_segment = duration(initial_q ,hold_time, model)

    # 拼接完整轨迹
    initial_trajectory = np.vstack((trajectory, hold_segment))  # shape: (N, 7)

    return initial_trajectory


def movement(start_q, target_q,
             model,
             hold=False,
             steps=200):
    trajectory = np.linspace(start_q[:7], target_q[:7], steps)  # zero -> initial


    if hold:
        hold_time = 3.0
        hold_segment = duration(target_q[:7], hold_time, model)

        # 拼接完整轨迹
        finial_trajectory = np.vstack((trajectory, hold_segment))  # shape: (N, 7)
    else:
        finial_trajectory = trajectory

    return finial_trajectory


def trajectory_show(trajectory, model, data, viewer):
    for q in trajectory:
        data.ctrl[:7] = q[:7]
        mujoco.mj_forward(model, data)

        # 每个姿态执行 5 个仿真步骤保持流畅
        for _ in range(30):
            mujoco.mj_step(model, data)
            viewer.sync()


# 解出一个展示一个
# def process_eeg_trajectory(eeg_data, model, model_data):
#     start_q, target_quat = initial_location(model, model_data)  # 初始位置和朝向
#     target_quat_xyzw = np.roll(target_quat, 1)
#
#     site_id = model.site("ee_site").id
#     current_q = start_q.copy()
#
#     for i, segment in enumerate(sliding_window(eeg_data, window_size=250, stride=250)):
#         # Step 1: 解码生成目标位置
#         target_pos = decode_target(segment)
#         print(f"[{i}] Target Position:", target_pos)
#
#         # Step 2: 利用 IK 获取目标位姿的关节角
#         q_target = ik_jacobian(target_pos, target_quat, site_id)
#         q_target[6] = current_q[6]  # 保持某些关节不变（如你已有代码）
#
#         # Step 3: 插值并显示
#         traj = movement(current_q, q_target)
#         trajectory_show(traj)
#
#         # 更新当前位姿
#         current_q = q_target.copy()


def target_compute(positions):
    thumb = positions[:, 0:3]  # shape: (N, 3)
    index = positions[:, 3:6]  # shape: (N, 3)

    midpoint = (thumb + index) / 2  # shape: (N, 3)

    targets = map_workspace(midpoint)

    print(targets)
    return targets


def sample_kin_points(kin, start, step):
    # 确保 kin 是 (6, 4389)
    assert kin.ndim == 2 and kin.shape[0] == 6

    end = kin.shape[1]

    # 起始点之后的索引列表：200, 450, 700, ..., up to < end
    indices = list(range(start, end, step))

    # 如果最后一个点不足 step 个也要取
    if indices[-1] != end - 1:
        indices.append(end - 1)

    # 取样
    sampled = kin[:, indices]

    return sampled


def draw_trajectory(targets, bounds_min=(0, 0, 0), bounds_max=(0.8, 0.8, 0.8)):
    """
    显示目标点轨迹线
    :param targets: shape (N, 3) 的 numpy 数组，按时间顺序排列的目标点坐标
    :param bounds_min: 3D 控件空间的最小边界
    :param bounds_max: 3D 控件空间的最大边界
    """
    targets = np.array(targets)  # Ensure it's an ndarray
    x, y, z = targets[:, 0], targets[:, 1], targets[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制轨迹线
    ax.plot(x, y, z, color='blue', label='Trajectory')
    # 绘制起点和终点
    ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='End')

    # 设置坐标轴范围
    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    ax.set_zlim(bounds_min[2], bounds_max[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Target Trajectory")
    ax.legend()
    ax.view_init(elev=30, azim=135)  # 可调视角

    plt.tight_layout()
    plt.show()


def targets_get(have_emg=False, output_kin=False):
    # EEG/ EEG + EMG
    if output_kin:
        # KIN
        kin = trial_data(have_emg=have_emg, kin_output=output_kin)
        sampled_kin=sample_kin_points(kin, start=200, step=250) # (6, S_T)
        targets = target_compute(sampled_kin.T)
        return targets
    else:
        if have_emg:
            eeg, emg =  trial_data(have_emg=have_emg, kin_output=False)  # input eeg in one trial
            positions = EEG_Decoder(eeg, emg, have_emg=have_emg)
            targets = target_compute(positions)

        else:
            trial = trial_data(have_emg=have_emg, kin_output=False)  # input eeg in one trial
            positions = EEG_Decoder(trial)
            targets = target_compute(positions)

        return targets

    # draw_trajectory(targets)

def main():
    # Mujoco
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    have_emg=True
    output_kin=False

    targets = targets_get(have_emg, output_kin)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            start_q, target_quat, trajectory_i = initial_location(model, data)

            # target_quat_xyzw = np.roll(target_quat, 1) #初始姿态的四元数组

            # 准备轨迹拼接容器
            full_trajectory = [*trajectory_i]  # 起始轨迹

            # 设置起点为当前位置
            current_q = start_q

            site_id = model.site("ee_site").id

            # 遍历所有目标点，逐个生成轨迹并拼接
            for target in targets:
                q_target = ik_jacobian_position(target, site_id, model, data)

                # 从当前关节角度到目标的轨迹
                trajectory = movement(current_q, q_target, model, steps=50)

                full_trajectory.extend(trajectory)

                current_q = q_target  # 更新当前关节角度为下次轨迹起点

            final_trajectory = movement(current_q, start_q, model, steps=50)
            full_trajectory.extend(final_trajectory)

            # 显示整条轨迹
            trajectory_show(np.array(full_trajectory), model, data, viewer)





            # # 单目标点轨迹示例
            # target = np.array([0.5, 0.6, 0.3])
            # # 计算逆运动学，求机械臂关节角度
            # end_effector_id = find_end_site(model)
            # site_id = model.site("ee_site").id
            # q_target = ik_jacobian_position(target, site_id, model, data)
            # # q_target[6]=start_q[6]
            #
            # trajectory = movement(start_q,q_target, model)
            # full_trajectory = np.vstack((trajectory_i, trajectory))
            # trajectory_show(full_trajectory, model, data, viewer)

            # # 在 viewer 中保持该姿态 5 秒（假设约 60FPS）
            # for _ in range(300):
            #     viewer.sync()

            # 初始化轨迹生成
            # trajectory = initial(model,data)
            # trajectory_show(trajectory, model, data, viewer)


if __name__ == "__main__":
    main()
    # have_emg = True
    # output_kin = False
    #
    # targets = targets_get(have_emg, output_kin)
    #
    # draw_trajectory(targets)






