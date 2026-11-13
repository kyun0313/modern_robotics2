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

MODE = "force"

# dt=0.002
phy_dt = 0.0001

def read_floor_force(model, data, floor_id, tip_id):
    total_f = 0.0
    contact_f = np.zeros(6)

    for i in range(data.ncon):
        contact = data.contact[i]

        is_tip_floor = ((contact.geom1 == floor_id and contact.geom2 == tip_id) or (contact.geom1 == tip_id and contact.geom2 == floor_id))

        if is_tip_floor:
            mujoco.mj_contactForce(model, data, i, contact_f)
            total_f += contact_f[0]

    return total_f

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



def main(task="setpoint", friction_on=False, seconds=10.0):
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    # model = mujoco.MjModel.from_xml_path("/home/marchlab/mujoco_env/mjctrl-main/franka_emika_panda/scene.xml")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "franka_emika_panda" / "scene_contact.xml"))
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = 0.0 #float(gravity_compensation)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0

    floor_id = model.geom("floor").id
    tip_id = model.geom("contact_tip").id

    model.geom_contype[[floor_id, tip_id]] = 1
    model.geom_conaffinity[[floor_id, tip_id]] = 1
    model.geom_condim[[floor_id, tip_id]] = 1

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

    gravity_data = mujoco.MjData(model)

    rows = []

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        q_start_deg = [0, 41, 0, -137, -1, 179, -44]
        data.qpos[:] = np.deg2rad(q_start_deg)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0

        mujoco.mj_forward(model, data)

        p_start = data.site_xpos[site_id].copy()
        R_ref = data.site_xmat[site_id].reshape(3, 3).copy()

        force_integral = 0.0
        F_filtered = 0.0

        z_command = p_start[2]
        vz_command = 0.0

        #data.ctrl[actuator_ids] = 0.0
        #mujoco.mj_forward(model, data)

        force_integral = 0.0

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        next_print = 0.0

        while viewer.is_running() and data.time < seconds:
            start = time.time()

            mujoco.mj_forward(model, data)
            p = data.site_xpos[site_id].copy()
            R = data.site_xmat[site_id].reshape(3, 3).copy()
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)

            v = jac_pos @ data.qvel
            omega = jac_rot @ data.qvel

            F = read_floor_force(model, data, floor_id, tip_id)

            # Fd
            if data.time < 3.0:
                Fd = 3.0 * min(data.time / 1.0, 1.0)
            elif data.time < 6.0:
                Fd = 5.0
            else:
                Fd = 3.0

            p_ref = p_start.copy()
            p_ref[2] = 0.025
            v_ref = np.zeros(3)

            # 제어기
            F_task = 500.0 * (p_ref - p) + 40.0 * (v_ref - v)

            f_e = Fd - F
            force_integral = np.clip(force_integral + f_e * dt, -5.0, 5.0,)

            press = np.clip(Fd + 0.2 * f_e + 2.0 * force_integral, 0.0, 12.0,)

            F_task[2] = -press - 15.0 * v[2]

            # end-effector 방향 유지
            rot_e = np.zeros(3)
            for i in range(3):
                rot_e += np.cross(R[:, i], R_ref[:, i])
            rot_e *= 0.5

            moment = 20.0 * rot_e - 4.0 * omega

            # 초기 중력 토크
            gravity_data.qpos[:] = data.qpos
            gravity_data.qvel[:] = 0.0
            mujoco.mj_forward(model, gravity_data)
            gravity = gravity_data.qfrc_bias.copy()

            tau = (gravity + jac_pos.T @ F_task + jac_rot.T @ moment - 0.2 * data.qvel)

            data.ctrl[actuator_ids] = np.clip(tau[dof_ids], model.actuator_ctrlrange[actuator_ids, 0], model.actuator_ctrlrange[actuator_ids, 1],)

            rows.append([data.time,Fd,F,F_filtered,p_ref[0],p[0],p_ref[2],p[2],z_command,])

            for _ in range(substeps):
                data.qfrc_applied[:] = 0.0

                if friction_on:
                    resistance = np.asarray(compute_joint_friction(data.qvel[dof_ids])).reshape(7)
                    data.qfrc_applied[dof_ids] = -resistance

                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(
                max(0.0, dt - (time.time() - start))
            )

    if rows:
        log = np.asarray(rows)
        t = log[:, 0]

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        force_label = (
            "Nominal load"
            if MODE in ("impedance", "admittance")
            else "Target"
        )

        axes[0].plot(t, log[:, 1], "--", label=force_label)
        axes[0].plot(t, log[:, 2], label="Measured contact force")
        axes[0].set_ylabel("Normal force [N]")

        axes[1].plot(t, 1000 * log[:, 4], "--", label="Reference X")
        axes[1].plot(t, 1000 * log[:, 5], label="Actual X")
        axes[1].set_ylabel("X [mm]")

        axes[2].plot(t, 1000 * log[:, 6], "--", label="Reference Z")
        axes[2].plot(t, 1000 * log[:, 7], label="Actual Z")

        if MODE == "admittance":
            axes[2].plot(t, 1000 * log[:, 8], ":", label="Admittance command Z")

        axes[2].set_ylabel("Z [mm]")
        axes[2].set_xlabel("Time [s]")

        for ax in axes:
            ax.grid(True)
            ax.legend()

        condition = "ON" if friction_on else "OFF"
        name = f"{MODE}_{task}_friction_{condition}"

        fig.suptitle(name)
        fig.tight_layout()

        output = ROOT / "results"
        output.mkdir(exist_ok=True)

        fig.savefig(output / f"{name}.png",dpi=300,bbox_inches="tight",)
        np.savez(output / f"{name}.npz",t=t,force_target=log[:, 1],force_measured=log[:, 2],force_filtered=log[:, 3],x_reference=log[:, 4],x_actual=log[:, 5],z_reference=log[:, 6],z_actual=log[:, 7],z_command=log[:, 8],)

        print("Last measured force [N]:", log[-1, 2])
        print("Saved:", output / f"{name}.png")
        plt.show()

            
            
if __name__ == "__main__":
    main(task="setpoint", friction_on=True, seconds=10.0,)
