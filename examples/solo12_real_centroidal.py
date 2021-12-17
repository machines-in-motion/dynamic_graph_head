import numpy as np

from dynamic_graph_head import ThreadHead, Vicon, HoldPDController

import time

import dynamic_graph_manager_cpp_bindings
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config

import mim_control_cpp

import pinocchio as pin

class CentroidalController:
    def __init__(self, head, vicon_name, mu, kp, kd, kc, dc, kb, db):
        self.set_k(kp, kd)
        self.robot = Solo12Config.buildRobotWrapper()
        self.vicon_name = vicon_name

        self.x_com = [0.0, 0.0, 0.20]
        self.xd_com = [0.0, 0.0, 0.0]

        self.x_des = np.array([
             0.2, 0.142, 0.015,  0.2, -0.142,  0.015,
            -0.2, 0.142, 0.015, -0.2, -0.142,  0.015
        ])
        self.xd_des = np.array(4*[0., 0., 0.])

        self.x_ori = [0., 0., 0., 1.]
        self.x_angvel = [0., 0., 0.]
        self.cnt_array = 4 * [1,]

        self.w_com = np.zeros(6)

        q_init = np.zeros(19)
        q_init[7] = 1
        self.centrl_pd_ctrl = mim_control_cpp.CentroidalPDController()
        self.centrl_pd_ctrl.initialize(2.5, np.diag(self.robot.mass(q_init)[3:6, 3:6]))

        self.force_qp = mim_control_cpp.CentroidalForceQPController()
        self.force_qp.initialize(4, mu, np.array([5e5, 5e5, 5e5, 1e6, 1e6, 1e6]))

        root_name = 'universe'
        endeff_names = ['FL_ANKLE', 'FR_ANKLE', 'HL_ANKLE', 'HR_ANKLE']
        self.imp_ctrls = [mim_control_cpp.ImpedanceController() for eff_name in endeff_names]
        for i, c in enumerate(self.imp_ctrls):
            c.initialize(self.robot.model, root_name, endeff_names[i])

        self.kc = np.array(kc)
        self.dc = np.array(dc)
        self.kb = np.array(kb)
        self.db = np.array(db)

        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')
        self.slider_positions = head.get_sensor('slider_positions')
        self.imu_gyroscope = head.get_sensor('imu_gyroscope')

    def set_k(self, kp, kd):
        self.kp = 4 * [kp, kp, kp, 0, 0, 0]
        self.kd = 4 * [kd, kd, kd, 0, 0, 0]

    def warmup(self, thread):
        thread.vicon.bias_position(self.vicon_name)
        self.zero_sliders = self.slider_positions.copy()

    def get_base(self, thread):
        base_pos, base_vel = thread.vicon.get_state(self.vicon_name)
        base_vel[3:] = self.imu_gyroscope
        return base_pos, base_vel

    def run(self, thread):
        base_pos, base_vel = self.get_base(thread)

        self.q = np.hstack([base_pos, self.joint_positions])
        self.dq = np.hstack([base_vel, self.joint_velocities])

        self.w_com[:] = 0

        self.centrl_pd_ctrl.run(
            self.kc, self.dc, self.kb, self.db,
            self.q[:3], self.x_com, self.dq[:3], self.xd_com,
            self.q[3:7], self.x_ori, self.dq[3:6], self.x_angvel
        )

        self.w_com[2] = 9.81 * Solo12Config.mass
        self.w_com += self.centrl_pd_ctrl.get_wrench()

        # distrubuting forces to the active end effectors
        pin_robot = self.robot
        pin_robot.framesForwardKinematics(self.q)
        com = self.com = pin_robot.com(self.q)
        rel_eff = np.array([
            pin_robot.data.oMf[i].translation - com for i in Solo12Config.end_eff_ids
        ]).reshape(-1)

        ext_cnt_array = [1., 1., 1., 1.]
        self.force_qp.run(self.w_com, rel_eff, ext_cnt_array)
        self.F = self.force_qp.get_forces()

        # passing forces to the impedance controller
        self.tau = np.zeros(18)
        for i, c in enumerate(self.imp_ctrls):
            c.run(self.q, self.dq,
                 np.array(self.kp[6*i:6*(i+1)]),
                 np.array(self.kd[6*i:6*(i+1)]),
                 1.,
                 pin.SE3(np.eye(3), np.array(self.x_des[3*i:3*(i+1)])),
                 pin.Motion(self.xd_des[3*i:3*(i+1)], np.zeros(3)),
                 pin.Force(self.F[3*i:3*(i+1)], np.zeros(3))
             )

            self.tau += c.get_torques()

        head.set_control('ctrl_joint_torques', self.tau[6:])

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

