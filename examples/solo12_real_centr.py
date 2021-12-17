import numpy as np

from dynamic_graph_head import ThreadHead, Vicon, HoldPDController

import time

import dynamic_graph_manager_cpp_bindings
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config

import mim_control_cpp
from mim_control_cpp import CentroidalImpedanceController

import pinocchio as pin

class CentroidalController:
    def __init__(self, head, vicon_name, mu, kp, kd, kc, dc, kb, db):
        self.set_k(kp, kd)
        self.robot = Solo12Config.buildRobotWrapper()
        self.robot_config = Solo12Config()
        self.vicon_name = vicon_name

        qp_penalty_weights = np.array([5e5, 5e5, 5e5, 1e6, 1e6, 1e6])

        q_init = self.robot_config.q0.copy()
        q_init[0] = 0.

        self.ctrl = CentroidalImpedanceController()
        self.ctrl.initialize(
            2.5,
            np.diag(self.robot.mass(q_init)[3:6, 3:6]),
            self.robot.model,
            "universe",
            self.robot_config.end_effector_names,
            mu,
            qp_penalty_weights,
            kc, dc, kb, db,
            self.kp, self.kd
        )

        # Desired center of mass position and velocity.
        self.x_com = [0.0, 0.0, 0.225]
        self.xd_com = [0.0, 0.0, 0.0]
        # The base should be flat.
        self.x_ori = [0.0, 0.0, 0.0, 1.0]
        self.x_angvel = [0.0, 0.0, 0.0]

        # Desired leg length
        self.x_des = [
             0.195,  0.147, 0.015, 0, 0, 0, 1.,
             0.195, -0.147, 0.015, 0, 0, 0, 1.,
            -0.195,  0.147, 0.015, 0, 0, 0, 1.,
            -0.195, -0.147, 0.015, 0, 0, 0, 1.
        ]
        self.xd_des = np.zeros(4 * 6)

        self.tau = np.zeros(12)
        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')
        self.imu_gyroscope = head.get_sensor('imu_gyroscope')

    def set_k(self, kp, kd):
        self.kp = [kp, kp, kp, 0, 0, 0]
        self.kd = [kd, kd, kd, 0, 0, 0]

    def warmup(self, thread):
        thread.vicon.bias_position(self.vicon_name)

    def get_base(self, thread):
        base_pos, base_vel = thread.vicon.get_state(self.vicon_name)
        base_vel[3:] = self.imu_gyroscope
        return base_pos, base_vel

    def run(self, thread):
        base_pos, base_vel = self.get_base(thread)

        self.q = np.hstack([base_pos, self.joint_positions])
        self.dq = np.hstack([base_vel, self.joint_velocities])

        quat = pin.Quaternion(self.q[6], self.q[3], self.q[4], self.q[5])

        self.ctrl.run(
            self.q, self.dq,
            np.array([1., 1., 1., 1.]),
            self.q[:3],
            self.x_com,
            quat.toRotationMatrix().dot(self.dq[:3]), # local to world frame
            self.xd_com,
            self.q[3:7],
            self.x_ori,
            self.dq[3:6],
            self.x_angvel,
            self.x_des,
            self.xd_des
        )

        self.tau[:] = self.ctrl.get_joint_torques()
        head.set_control('ctrl_joint_torques', self.tau)

print("Finished imports")

###
# Create the dgm communication and instantiate the controllers.
head = dynamic_graph_manager_cpp_bindings.DGMHead(Solo12Config.dgm_yaml_path)

# Create the controllers.
hold_pd_controller = HoldPDController(head, 3., 0.05, with_sliders=True)

ctrl = centroidal_controller = CentroidalController(head, 'solo12/solo12', 0.2, 50., 0.7,
    [100., 100., 100.], [15., 15., 15.], [25., 25., 25.], [22.5, 22.5, 22.5]
)

thread_head = ThreadHead(
    0.001,
    hold_pd_controller,
    head,
    [
        ('vicon', Vicon('172.24.117.119:801', ['solo12/solo12']))
    ]
)

# Start the parallel processing.
thread_head.start()

def go_hold():
    thread_head.switch_controllers(hold_pd_controller)

def go_cent():
    thread_head.switch_controllers(ctrl)

