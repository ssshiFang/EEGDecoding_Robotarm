import mujoco #物理建模和核心参数
import mujoco.viewer #提供GUI支持，用于实时模拟场景
import numpy as np

model = mujoco.MjModel.from_xml_path("eeg_arm.xml")
data = mujoco.MjData(model) #仿真时用来保存模型的类

# 简单的 EEG 模拟：返回一个归一化的 [x, y, z]
def decode_eeg():
    # 可替换为你的脑电接口
    t = data.time
    return [
        0.5 + 0.4 * np.sin(t),
        0.5 + 0.4 * np.cos(t),
        0.5 + 0.4 * np.sin(2 * t)
    ]

# 映射 EEG 坐标到工作空间坐标
def map_eeg_to_target(eeg_pos):
    workspace_min = np.array([0.1, -0.2, 0.0])
    workspace_max = np.array([0.4,  0.2, 0.0])  # 平面控制，z 暂不用于 IK
    return workspace_min + eeg_pos * (workspace_max - workspace_min)

# 简单逆运动学：暴力搜索解决（适用于 3-DOF 臂）
def simple_ik(target, max_iter=1000):
    best_q = data.qpos.copy()
    best_dist = float('inf')

    for _ in range(max_iter):
        q = np.random.uniform(low=-np.pi, high=np.pi, size=3)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        ee_pos = data.site("ee_site").xpos
        dist = np.linalg.norm(ee_pos - target)
        if dist < best_dist:
            best_dist = dist
            best_q = q.copy()

    return best_q

# GUI 主循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        eeg_pos = np.array(decode_eeg())  # EEG 模拟输入
        target = map_eeg_to_target(eeg_pos)  # EEG → 空间坐标
        q_target = simple_ik(target)  # 空间坐标 → 关节角度

        data.qpos[:] = q_target
        mujoco.mj_forward(model, data) #执行物理仿真步，更新位置速度
        mujoco.mj_step(model, data)
        viewer.sync() #更新后的状态显示在窗口上，刷新GUI


