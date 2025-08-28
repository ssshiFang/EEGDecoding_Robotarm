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
from r_model.model250_emb256_h8 import EEGTransformerModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import imageio
import pickle
import os


def align(eeg, emg, delay=200):
    assert eeg.shape[1] == emg.shape[1], "EEG and EMG should have same length"
    time_steps = eeg.shape[1]

    if delay >= time_steps:
        raise ValueError("delay not over length")

    new_len = time_steps - delay
    eeg_aligned = eeg[:, :new_len]
    emg_aligned = emg[:, delay:]

    return eeg_aligned, emg_aligned


def trial_data(have_emg= False, kin_output=False):
    # current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # child document
    save_dir = os.path.join(parent_dir, 'electrical_arm')

    # data save path
    eeg_path = os.path.join(save_dir, 'trial_data/4_emg_arm_trial1.pkl')

    with open(eeg_path, 'rb') as f:
        trial = pickle.load(f)


    if kin_output:
        kin_trial = trial[0]['kin']  # shape: (6, T)
        return kin_trial

    if have_emg:
        eeg_trial = trial[0]['eeg']  # shape: (32, T)
        emg_trial = trial[0]['emg']  # shape: (N_emg, T)
        eeg, emg = align(eeg_trial, emg_trial)
        return eeg, emg

    else:
        eeg_trial = trial[0]['eeg']  # shape: (32, T)
        return eeg_trial





# EEG docoding test
def EEG_Decoder(eeg_data, emg_data=None, segment_length=250, have_emg=False):
    """
    Using model for decoding
    eeg_data: (32, T)  have_emg= True emg (5, T)
    segment_length windows of eeg and kin, emg
    return: (N, 6)
    """

    # current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # model directory
    model_path = os.path.join(parent_dir, 'f_model/4_eeg_arm_best_model0.88.pth')

    # device choose
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
        last_chunk = eeg_data[:, -segment_length:]  # last part 250
        segments.append(last_chunk)

    # change to batch tensor
    eeg_batch = torch.tensor(np.stack(segments), dtype=torch.float32).to(device)  # (N, 32, 250)

    if have_emg:
        C, T = emg_data.shape
        seg = []

        for i in range(T // segment_length):
            cut = emg_data[:, i * segment_length:(i + 1) * segment_length]
            seg.append(cut)

        if T % segment_length != 0:
            last_cut = emg_data[:, -segment_length:]  # last part 250
            seg.append(last_cut)

        # change to batch tensor
        emg_batch = torch.tensor(np.stack(seg), dtype=torch.float32).to(device)  # (N, 32, 250)

        with torch.no_grad():
            outputs = model(eeg_batch, emg_batch)  # (N, 6)
            positions = outputs.cpu().numpy()

        return positions


    # decoding

    with torch.no_grad():
        outputs = model(eeg_batch)  # (N, 6)
        positions = outputs.cpu().numpy()

    return positions


#  'hand' chech the end-eff body ID
def find_end_site(model):
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    if end_effector_id == -1:
        # if not have end-eff
        print("Warning: Could not find the end effector with the given name.")

    return end_effector_id


def limit_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

# normalize to robot arm work place
def map_workspace(coords):
    coords[:, 1] = 1 - coords[:, 1] # y 1-0 -> 0-1
    workspace_min = np.array([0.3, 0.03, 0.2])
    workspace_max = np.array([0.7, 0.6, 0.55])
    return workspace_min + coords * (workspace_max - workspace_min)

#simple IK jacobian matrix
def ik_jacobian(target_pos,
                target_quat,
                site_id,
                model,
                data,
                max_iters=100, # repeat time
                tol=1e-3, # tolerance
                alpha=0.1): # step
    q = data.qpos[:].copy() # site location

    # point to initial
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
        # quat for pose
        mujoco.mju_subQuat(res, target_quat, ee_quat)
        quat_err = res  # axis-angle

        # location+ pos loss
        error = np.concatenate([pos_err, quat_err])

        print(f" Error Norm: {np.linalg.norm(error):.4f}")
        print(f"EE Pos: {ee_pos}")

        if np.linalg.norm(error) < tol: # distance of two points
            break

        # jacobian matrix
        jacp = np.zeros((3, model.nv)) # IK运动控制 location
        jacr = np.zeros((3, model.nv)) # 姿态控制
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        # 6xN jacobian matrix
        jac_full = np.vstack([jacp, jacr])[:, :model.nv]

        # # freeze 7joint index- 5 set0
        # jac_full[:, 6] = 0

        # IK update
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

        ee_pos = data.site(site_id).xpos  # end-eff location
        pos_err = target_pos - ee_pos     # location loss

        # print(f"Error Norm: {np.linalg.norm(pos_err):.4f}")
        # print(f"EE Pos: {ee_pos}")

        if np.linalg.norm(pos_err) < tol:
            break

        # jacobian
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)

        # update joint
        dq = alpha * np.linalg.pinv(jacp[:, :model.nv]) @ pos_err
        q[:model.nv] += dq

    return q


def initial_location(model,
                     data,
                     hold_time=1.0,
                     steps=40):
    start_q=numpy.zeros(7)
    initial_q=numpy.array([0,0.1,0,-3,2.9,1.65,-2.4]) # initial [0,0.335,0,-3,2.9,1.4,-2.4,0.02,0.02]
    # initial_q = numpy.array([-0.78, 0.1, 0, -3, 2.9, 1.65, -2.4])
    # range (x,y 0.3 -0.78   0.24 - 0.55 z 0.12 - 0.65)
    # xy equal -0.782
    # 2 range -0.4 - 1.55
    # 3 range -1.8 - 1.5
    # 5 range 2 - 2.9
    # end site location [0.31431234 0.03086261 0.12288893]

    initial_q = initial_q.copy()
    # joint location
    data.qpos[:7] = initial_q

    mujoco.mj_forward(model, data)

    site_id = model.site("ee_site").id
    initial_xmat = data.site(site_id).xmat.reshape(3, 3)
    target_quat = R.from_matrix(initial_xmat).as_quat()

    trajectory = np.linspace(start_q, initial_q[:7], steps)  # zero -> initial
    hold_segment = duration(initial_q, hold_time, model)

    # all trajectory
    initial_trajectory = np.vstack((trajectory, hold_segment))  # shape: (N, 7)

    # pint location
    # end_effector_id = find_end_site()
    # end_effector_pos = data.body(end_effector_id).xpos
    # print("Actuators in model:", model.nu)
    # print(end_effector_pos)
    # print("nq =", model.nq)
    # print("qpos0 =", model.qpos0)
    # print("initial_q = ", data.qpos[:])
    # for i in range(model.njnt):
    #     print(f"Joint {i}: name = {model.joint(i).name}, type = {model.joint(i).type}, addr = {model.jnt_qposadr[i]}")

    # mujoco.mj_step(model, data)simulation mujoco.mj_forward(model, data)get outcome

    return initial_q, target_quat, initial_trajectory


def generate_joint_trajectory(start, goal, steps):
    return np.linspace(start, goal, steps)


# hold part
def duration(target_q, time_len, model):
    hold_duration = 2.0  # hold time
    sim_freq = int(1 / model.opt.timestep)  #  500~1000 Hz
    hold_steps = int(time_len * sim_freq / 20)  # decrease to viewer viusal
    hold_segment = np.tile(target_q, (hold_steps, 1))  # hold

    return hold_segment


def initial(model,data,
            hold_time=1.0,
            steps=500):
    start_q=numpy.zeros(7)
    initial_q=numpy.array([-0.78,0.15,0,-3,2.9,1.65,-2.4]) # initial [0,0.335,0,-3,2.9,1.4,-2.4]

    trajectory = np.linspace(start_q, initial_q, steps) # zero -> initial
    hold_segment = duration(initial_q ,hold_time, model)

    # get together
    initial_trajectory = np.vstack((trajectory, hold_segment))  # shape: (N, 7)

    return initial_trajectory


def movement(start_q, target_q,
             model,
             hold=False,
             steps=200):
    trajectory = np.linspace(start_q[:7], target_q[:7], steps)  # initial -> target


    if hold:
        hold_time = 3.0
        hold_segment = duration(target_q[:7], hold_time, model)

        # get together
        finial_trajectory = np.vstack((trajectory, hold_segment))  # shape: (N, 7)
    else:
        finial_trajectory = trajectory

    return finial_trajectory


def trajectory_show(trajectory, model, data, viewer, save_gif, save_path):
    frames = []

    renderer = mujoco.Renderer(model)

    for q in trajectory:
        data.ctrl[:7] = q[:7]
        mujoco.mj_forward(model, data)

        # step show
        for _ in range(30):
            mujoco.mj_step(model, data)
            viewer.sync()

            if save_gif:
                try:
                    renderer.update_scene(data)
                    frame = renderer.render()
                    frames.append(frame)
                except Exception as e:
                    print("error:", e)

    if save_gif and len(frames) > 0:
        imageio.mimsave(save_path, frames, fps=15)


def target_compute(positions):
    thumb = positions[:, 0:3]  # shape: (N, 3)
    index = positions[:, 3:6]  # shape: (N, 3)

    midpoint = (thumb + index) / 2  # shape: (N, 3)

    targets = map_workspace(midpoint)

    print(targets)
    return targets


def sample_kin_points(kin, start, step):
    # kin type (6, T)
    assert kin.ndim == 2 and kin.shape[0] == 6

    end = kin.shape[1]

    # index 200, 450, 700, ..., up to < end
    indices = list(range(start, end, step))

    # last one get
    if indices[-1] != end - 1:
        indices.append(end - 1)

    # 取样
    sampled = kin[:, indices]

    return sampled


def draw_trajectory(targets, bounds_min=(0, 0, 0), bounds_max=(0.8, 0.8, 0.8), output=False):
    """
    Show trajectory
    targets: (N, 3)
    """
    targets = np.array(targets)  # Ensure it's an ndarray
    x, y, z = targets[:, 0], targets[:, 1], targets[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.plot(x, y, z, color='blue', label='Trajectory')

    ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='End')

    # location range
    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    ax.set_zlim(bounds_min[2], bounds_max[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Target Trajectory")
    # ax.set_title("Participant 9 EEG")
    # ax.set_title("Participant 9 EEG EMG Multi-modality")
    # ax.set_title("Participant 9 Kinematic Trajectory")
    ax.legend()
    ax.view_init(elev=20, azim=-45)  # angle view

    if output == True:
        # 更新函数
        def update(angle):
            ax.view_init(elev=30, azim=angle)  # elev-z, azim
            return fig,

        # gif
        ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 240), interval=50, blit=False)

        ani.save('arotation.gif', writer='pillow', fps=10)

        plt.close(fig)

    else:
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
            eeg, emg = trial_data(have_emg=have_emg, kin_output=False)  # input eeg in one trial
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
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # gif save path
    save_path = os.path.join(current_dir, 'gif/4_emg_best_model0.96_t250_s50_w200.gif')

    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    have_emg=False
    output_kin=False

    targets = targets_get(have_emg, output_kin)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD # show x y z
        while viewer.is_running():
            start_q, target_quat, trajectory_i = initial_location(model, data)

            # target_quat_xyzw = np.roll(target_quat, 1) #quat of inital pose

            # all trajectory
            full_trajectory = [*trajectory_i]  # start

            # set location
            current_q = start_q

            site_id = model.site("ee_site").id

            # all target trajectory
            for target in targets:
                q_target = ik_jacobian_position(target, site_id, model, data)

                # form current join angle to target point
                trajectory = movement(current_q, q_target, model, steps=50)

                full_trajectory.extend(trajectory)

                current_q = q_target  # update

            # trajectory for come into initial
            # final_trajectory = movement(current_q, start_q, model, steps=50)
            # full_trajectory.extend(final_trajectory)

            # show trajectory
            trajectory_show(np.array(full_trajectory), model, data, viewer, save_gif=False, save_path=save_path)

            # # Example of a single target point trajectory
            # target = np.array([0.5, 0.6, 0.3])
            # # Compute inverse kinematics to get robot joint angles
            # end_effector_id = find_end_site(model)
            # site_id = model.site("ee_site").id
            # q_target = ik_jacobian_position(target, site_id, model, data)
            # # Optionally, set the last joint to start value
            # # q_target[6] = start_q[6]

            # # Generate movement from start to target joint configuration
            # trajectory = movement(start_q, q_target, model)
            # full_trajectory = np.vstack((trajectory_i, trajectory))
            # trajectory_show(full_trajectory, model, data, viewer)

            # # Hold the end-effector at this pose in the viewer for ~5 seconds (assuming ~60FPS)
            # for _ in range(300):
            #     viewer.sync()

            # # Initialize trajectory generation
            # trajectory = initial(model, data)
            # trajectory_show(trajectory, model, data, viewer)


if __name__ == "__main__":
    main()
    # have_emg = False
    # output_kin = False
    #
    # targets = targets_get(have_emg, output_kin)
    #
    # draw_trajectory(targets, output=False)






