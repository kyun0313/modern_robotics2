import mujoco
import mujoco.viewer
import numpy as np
import time
import modern_robotics as mr

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

    np.savez(
        output / f"{name}.npz",
        t=t, target=target, actual=actual, error=error,
    )

    print("Axis RMSE:", np.sqrt(np.mean(error**2, axis=0)))
    print("Final errors:", error[-1])
    print("Saved:", output / f"{name}.png")

    plt.show()

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



def main(task="trajectory", friction_on=False, seconds=10.0):
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    # model = mujoco.MjModel.from_xml_path("/home/marchlab/mujoco_env/mjctrl-main/franka_emika_panda/scene.xml")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "franka_emika_panda" / "scene.xml"))
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = 0.0
    gravity_data = mujoco.MjData(model)
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
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

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
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    rows = []

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        mujoco.mj_forward(model, data)

        X0 = np.eye(4)
        X0[:3, :3] = data.site(site_id).xmat.reshape(3, 3)
        X0[:3, 3] = data.site(site_id).xpos

        move = np.array([0.1, -0.1, -0.2])
        duration = 5.0

        Kp = 4.0
        Ki = 0.5
        error_integral = np.zeros(6)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        next_print = 0.0

        while viewer.is_running() and data.time < seconds:
            step_start = time.time()

            mujoco.mj_forward(model, data)

            # 현재 자세 X
            X = np.eye(4)
            R = data.site(site_id).xmat.reshape(3, 3)
            X[:3, :3] = R
            X[:3, 3] = data.site(site_id).xpos

            # 목표자세 Xd, 목표속도 Vd
            
            Xd = X0.copy()
            Vd = np.zeros(6)

            if task == "setpoint":
                Xd[:3, 3] += move
            else:
                s = np.clip(data.time / duration, 0.0, 1.0)
                h = 10*s**3 - 15*s**4 + 6*s**5
                hdot = (30*s**2 - 60*s**3 + 30*s**4) / duration
                Xd[:3, 3] += move * h
                Vd[3:] = (Xd[:3, :3].T @ (move * hdot))

            # 자세 오차
            X_e = mr.TransInv(X) @ Xd
            e = mr.se3ToVec(mr.MatrixLog6(X_e))

            # 적분 제한
            error_integral += e * dt
            error_integral = np.clip(error_integral, -0.2, 0.2)

            # 11.3.3 controller
            twist = (mr.Adjoint(X_e) @ Vd + Kp*e + Ki*error_integral)

            # Jacobian.
            jac_world = np.zeros((6, model.nv))
            mujoco.mj_jacSite(model, data, jac_world[:3], jac_world[3:], site_id)

            R = data.site(site_id).xmat.reshape(3, 3)

            jac = np.vstack((R.T @ jac_world[3:], R.T @ jac_world[:3]))

            # Damped least squares.
            dq_task = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

            # Nullspace control biasing joint velocities towards the home configuration.
            dq_null = (eye - np.linalg.pinv(jac) @ jac) @ (Kn * (q0 - data.qpos[dof_ids]))

            dq_before_limit = dq_task + dq_null
            dq = dq_before_limit.copy()

            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq, integration_dt)
            np.clip(q, *model.jnt_range.T, out=q)

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = q[dof_ids]
            if data.time >= next_print:
                actual = data.site(site_id).xpos

                print(f"\n시간: {data.time:.1f}초")
                print("실행파일:", __file__)
                print("Kp, Ki:", Kp, Ki)
                print("목표 이동량:", np.round(Xd[:3, 3]-X0[:3, 3], 4), "실제 이동량:", np.round(actual-X0[:3, 3], 4))
                print("말단 속도 명령:", np.round(twist, 4))
                print("관절 속도 명령:", np.round(dq, 4))
                print("관절 속도 명령 태스크:", np.round(dq_task, 4))
                print("관절 속도 명령 널:", np.round(dq_null, 4))
                next_print += 1.0
            
            actual = data.site(site_id).xpos.copy()

            rows.append(np.r_[data.time, Xd[:3, 3], actual, Xd[:3, 3] - actual, e[:3], dq, data.qvel[dof_ids].copy()])

            for _ in range(substeps):
                data.qfrc_applied[:] = 0.0

                if gravity_compensation:
                    gravity_data.qpos[:] = data.qpos
                    gravity_data.qvel[:] = 0.0

                    mujoco.mj_forward(model, gravity_data)

                    data.qfrc_applied[:] = gravity_data.qfrc_bias

                if friction_on:
                    resistance = np.asarray(compute_joint_friction(data.qvel[dof_ids]), dtype=float,).reshape(7)
                    data.qfrc_applied[dof_ids] -= resistance
                
                mujoco.mj_step(model, data)

            draw_target(viewer, start=X0[:3, 3], goal=X0[:3, 3] + move, current_target=Xd[:3, 3], task=task,)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    if rows:
        log = np.asarray(rows)

        plot_tracking(t=log[:, 0], target=1000.0 * log[:, 1:4], actual=1000.0 * log[:, 4:7], labels=["End-effector X", "End-effector Y", "End-effector Z"], unit="Position [mm]", task=task, friction_on=friction_on,)
            
if __name__ == "__main__":
    main(task="setpoint", friction_on=False, seconds=10.0,)
