import mujoco
import mujoco.viewer
import numpy as np
import time
import modern_robotics as mr
import xml.etree.ElementTree as ET

from pathlib import Path
from datetime import datetime
import argparse

import matplotlib.pyplot as plt

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785

ROOT = Path(__file__).resolve().parent

# dt=0.002
phy_dt = 0.0001

def plot_tracking(t, target, actual, labels, unit, task, friction_on):
    error = target - actual
    count = target.shape[1]

    fig, axes = plt.subplots((count + 2) // 2, 2, figsize=(12, 3 * ((count + 2) // 2)), sharex=True,)
    axes = axes.ravel()

    for i in range(count):
        axes[i].plot(t, target[:, i], "--", label="Target")
        axes[i].plot(t, actual[:, i], label="Actual")
        axes[i].set_title(labels[i])
        axes[i].set_ylabel(unit)
        axes[i].grid(True)
        axes[i].legend()

    axes[count].plot(t, np.linalg.norm(error, axis=1))
    axes[count].set_title("Tracking error norm")
    axes[count].set_ylabel(unit)
    axes[count].grid(True)

    for ax in axes:
        ax.set_xlabel("Time [s]")

    condition = "ON" if friction_on else "OFF"
    name = f"{Path(__file__).stem}_{task}_friction_{condition}"
    fig.suptitle(name)
    fig.tight_layout()

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")

    np.savez(output / f"{name}.npz", t=t, target=target, actual=actual, error=error,)

    print("Axis RMSE:", np.sqrt(np.mean(error**2, axis=0)))
    print("Final errors:", error[-1])
    print("Saved:", output / f"{name}.png")

    plt.show()

def read_bias(model, work, q, dq):
    work.qpos[:] = q
    work.qvel[:] = dq
    mujoco.mj_forward(model, work)
    return work.qfrc_bias.copy()


def read_mass(model, data):
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    return M

def compute_joint_friction(dq):

        dq = np.asarray(dq).reshape(-1, 1)
        tau_f = np.zeros([len(dq), 1])

        if len(dq) == 7:
            F_c_n = [-15.876, -8.3559, -3.0788, -2.6335, -2.2146, -1.8977, -0.1]
            F_c_p = [15.693,  10.643,  2.6631,  2.4529,  2.1265,  1.8386, 0.1]
            F_v1_n = [-103.03, -71.122, -4.943,  -4.1702, -2.0895, -5.9460, -3]
            F_v1_p = [106.10,  86.132,  10.142,  4.1354,  2.2052,  5.4817, 3]
            F_v2_n = [-0.6523,  0.5446, -0.7078, -0.9260, 0.5379, -2.5524, -1]
            F_v2_p = [0.6846,  0.6741, -0.6764, -0.7183, 0.6657, -2.0797, 1]

            kc = 100
            tau_f
            for i in range(len(dq)):
                if dq[i]>0:
                    tau_f[i] = F_c_p[i]*np.tanh(kc*np.abs(dq[i])) + F_v1_p[i]*(1-np.exp(-np.abs(dq[i]/F_v2_p[i])))
                else:
                    tau_f[i] = F_c_n[i] * np.tanh(kc * np.abs(dq[i])) + F_v1_n[i] * (1 - np.exp(-np.abs(dq[i] / F_v2_n[i])))
            tau_f = np.diag([1, 1, 1, 0.5, 0.2, 0.01, 0.01]) @ tau_f
        elif len(dq) == 6:
            F_c_n = [-15.876, -8.3559, -3.0788, -2.6335, -2.2146, -1.8977]
            F_c_p = [15.693,  10.643,  2.6631,  2.4529,  2.1265,  1.8386]
            F_v1_n = [-103.03, -71.122, -4.943,  -4.1702, -2.0895, -5.9460]
            F_v1_p = [106.10,  86.132,  10.142,  4.1354,  2.2052,  5.4817]
            F_v2_n = [-0.6523,  0.5446, -0.7078, -0.9260, 0.5379, -2.5524]
            F_v2_p = [0.6846,  0.6741, -0.6764, -0.7183, 0.6657, -2.0797]

            kc = 50
            tau_f
            for i in range(len(dq)):
                if dq[i]>0:
                    tau_f[i] = F_c_p[i]*np.tanh(kc*np.abs(dq[i])) + F_v1_p[i]*(1-np.exp(-np.abs(dq[i]/F_v2_p[i])))
                else:
                    tau_f[i] = F_c_n[i] * np.tanh(kc * np.abs(dq[i])) + F_v1_n[i] * (1 - np.exp(-np.abs(dq[i] / F_v2_n[i])))
            tau_f = np.diag([1, 1, 1, 0.5, 0.2, 0.01]) @ tau_f
            
        return tau_f

def draw_target(viewer, start, goal, current_target, task):
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0

        def add_point(pos, rad, col):
            geom = scene.geoms[scene.ngeom]

            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0.0, 0.0]), np.asarray(pos, dtype=float), np.eye(3).ravel(), np.asarray(col, dtype=np.float32),)
            scene.ngeom += 1

        if task == "trajectory" and np.linalg.norm(goal - start) > 1e-9:
            geom = scene.geoms[scene.ngeom]

            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, np.asarray(start, dtype=float), np.asarray(goal, dtype=float),)
            scene.ngeom += 1

        add_point(goal, 0.006, [1.0, 0.1, 0.1, 1.0])

        if task == "trajectory":
            add_point(current_target, 0.004, [1.0, 0.9, 0.0, 1.0])



def main(task= "setpoint", friction_on=False, seconds=10.0):
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    # model = mujoco.MjModel.from_xml_path("/home/marchlab/mujoco_env/mjctrl-main/franka_emika_panda/scene.xml")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "franka_emika_panda" / "scene.xml"))
    data = mujoco.MjData(model)
    assert model.nq == model.nv == 7

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = 0.0 #float(gravity_compensation)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0

    # model.opt.timestep = dt
    model.opt.timestep = phy_dt
    substeps = round(dt/phy_dt)
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    joint_ids = np.array([model.joint(name).id for name in joint_names])
    dof_ids = model.jnt_dofadr[joint_ids]
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # change to torque input
    model.actuator_gaintype[actuator_ids] = (mujoco.mjtGain.mjGAIN_FIXED)
    model.actuator_biastype[actuator_ids] = (mujoco.mjtBias.mjBIAS_NONE)

    model.actuator_gainprm[actuator_ids, :] = 0.0
    model.actuator_gainprm[actuator_ids, 0] = 1.0
    model.actuator_biasprm[actuator_ids, :] = 0.0

    model.actuator_ctrlrange[actuator_ids] = (model.actuator_forcerange[actuator_ids])
    model.actuator_ctrllimited[actuator_ids] = True

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]
    target_id = model.body("target").id
    model.geom_rgba[model.geom_bodyid == target_id, 3] = 0.0
    model.site_rgba[model.site_bodyid == target_id, 3] = 0.0


    # Pre-allocate numpy arrays.
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))

    dynamics_data = mujoco.MjData(model)

    rows = []

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        data.ctrl[actuator_ids] = 0.0
        mujoco.mj_forward(model, data)

        force_integral = 0.0
        q_start = data.qpos.copy()

        # 목표 관절 이동 각 [rad]
        delta_q = np.array([0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1])

        Kp_pid = np.array([40.0, 40.0, 30.0, 30.0, 10.0, 8.0, 5.0])
        Kd_pid = np.array([12.0, 12.0, 8.0, 8.0, 2.0, 1.5, 1.0])
        Ki_pid = np.array([4.0, 4.0, 3.0, 3.0, 1.0, 0.8, 0.5])
        error_integral = np.zeros(7)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        next_print = 0.0

        while viewer.is_running() and data.time < seconds:
            start = time.time()
            mujoco.mj_forward(model, data)

            q = data.qpos.copy()
            dq = data.qvel.copy()

            # q_d, dq_d
            if task == "setpoint":
                q_d = q_start + delta_q
                dq_d = np.zeros(7)

            else:
                move_time = 5.0
                s = np.clip(data.time / move_time, 0.0, 1.0)

                h = 10*s**3 - 15*s**4 + 6*s**5
                dh = (30*s**2 - 60*s**3 + 30*s**4) / move_time

                q_d = q_start + delta_q*h
                dq_d = delta_q*dh

            # 오차
            e = q_d - q
            de = dq_d - dq

            g = read_bias(model, dynamics_data, q, np.zeros(7))

            # 11.38 식 구현
            tau = g + Kp_pid * e + Kd_pid * de + Ki_pid * error_integral

            data.ctrl[actuator_ids] = np.clip(tau[dof_ids], model.actuator_ctrlrange[actuator_ids, 0], model.actuator_ctrlrange[actuator_ids, 1],)

            rows.append([data.time, *q_d, *q])

            error_integral += e * dt

            for _ in range(substeps):
                data.qfrc_applied[:] = 0.0

                if friction_on:
                    resistance = np.asarray(compute_joint_friction(data.qvel[dof_ids])).reshape(7)
                    data.qfrc_applied[dof_ids] -= resistance

                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(
                max(0.0, dt - (time.time() - start))
            )

    if rows:
        log = np.asarray(rows)

        plot_tracking(t=log[:, 0], target=np.rad2deg(log[:, 1:8]), actual=np.rad2deg(log[:, 8:15]), labels=[f"Joint {i + 1}" for i in range(7)], unit="Angle [deg]", task=task, friction_on=friction_on,)

            
            
if __name__ == "__main__":
    main(task="setpoint", friction_on=True, seconds=300.0,)
