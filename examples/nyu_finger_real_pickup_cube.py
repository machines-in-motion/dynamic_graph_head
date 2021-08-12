# Author: Julian Viereck
# Date: Aug 12, 2021

import numpy as np
np.set_printoptions(precision=3, suppress=True)

from robot_properties_nyu_finger.config import (
    NYUFingerDoubleConfig0, NYUFingerDoubleConfig1)

from dynamic_graph_head import ThreadHead,  HoldPDController
import dynamic_graph_manager_cpp_bindings


class PDController:
  def __init__(self, head, Kp_gain, Kd_gain):
    self.head = head
    self.Kp_gain = Kp_gain
    self.Kd_gain = Kd_gain

    self.joint_positions = head.get_sensor('joint_positions')
    self.joint_velocities = head.get_sensor('joint_velocities')

  def warmup(self, thread_head):
    self.update_desired_pos(self.joint_positions)

  def update_desired_pos(self, new_target_pos):
    self.desired_pos = new_target_pos.copy()

  def run(self, thread_head):
    self.tau = (self.Kp_gain * (self.desired_pos - self.joint_positions) -
                self.Kd_gain * self.joint_velocities)
    self.head.set_control('ctrl_joint_torques', self.tau)


class PDIKController:
  def __init__(self, head, Kp_gain, Kd_gain):
    self.head = head

    self.L1 = 0.16
    self.L2 = 0.16

    self.q0 = -0.20

    self.Kp_gain = Kp_gain
    self.Kd_gain = Kd_gain

    self.joint_positions = head.get_sensor('joint_positions')
    self.joint_velocities = head.get_sensor('joint_velocities')

  def warmup(self, thread_head):
    self.update_desired_pos(self.joint_positions)

  def update_desired_pos(self, new_target_pos):
    self.desired_pos = new_target_pos.copy()

  def update_desired_pos_xy(self, new_target_pos_xy):
    # Use inverse kinematics to compute the new desired position.
    # NOTE: Need to swap axes to have orientation in right coordinate frames
    x_p = -new_target_pos_xy[1]
    y_p = -new_target_pos_xy[0]

    q2 = np.arccos((x_p**2 + y_p**2 - self.L1**2 - self.L2**2) / (2*self.L1*self.L2) )
    q1 = np.arctan2(y_p, x_p) + np.arctan2(self.L2 * np.sin(q2), self.L1 + self.L2 * np.cos(q2))
    self.desired_pos = np.array([0, q1, -q2])
    return self.desired_pos

  def run(self, thread_head):
    ti = thread_head.ti % 4000

    if ti in range(0, 1000):
      # [-0.05, -0.2] -> [0.0, -0.2]
      #
      # ti =    0 => x = -0.05
      # ti = 1000 => x = 0.00
      x, y = [-0.05 + 0.05 * ti / 1000, -0.3]
    elif ti in range(1000, 2000):
      #  [0.0, -0.2] -> [0.0, -0.1]
      #
      # ti = 1000 => y = -0.2
      # ti = 2000 => y = -0.1
      x, y = [0.00, -0.3 + 0.1 * (ti - 1000)/1000]
    elif ti in range(2000, 3000):
      # [0.0, -0.1] -> [0.0, -0.2]
      #
      # ti = 2000 => y = -0.1
      # ti = 3000 => y = -0.2
      x, y = [0.00, -0.2 + -0.1 * (ti - 2000)/1000]
    else:
      # [0.0, -0.2] -> [-0.05, -0.2]
      #
      # ti = 3000 => x = 0.05
      # ti = 4000 => x = -0.05
      x, y = [0.00 - 0.05 * (ti - 3000) / 1000, -0.3]

    self.update_desired_pos_xy([x, y])
    self.desired_pos[0] = self.q0

    self.tau = (self.Kp_gain * (self.desired_pos - self.joint_positions) -
                self.Kd_gain * self.joint_velocities)
    self.head.set_control('ctrl_joint_torques', self.tau)

###
# Create the dgm communication and instantiate the controllers.
head0 = dynamic_graph_manager_cpp_bindings.DGMHead(NYUFingerDoubleConfig0.dgm_yaml_path)
head1 = dynamic_graph_manager_cpp_bindings.DGMHead(NYUFingerDoubleConfig1.dgm_yaml_path)

# Create the controllers.
hold_pd_controller0 = HoldPDController(head0, 3., 0.05, with_sliders=False)
hold_pd_controller1 = HoldPDController(head1, 3., 0.05, with_sliders=False)

# Simple joint PD controller.
pd_controller0 = PDController(head0, 3., 0.05)
pd_controller1 = PDController(head1, 3., 0.05)

# Inverse Kinematics PD controller.
ik_ctrl0 = PDIKController(head0, 3., 0.05)
ik_ctrl1 = PDIKController(head1, 3., 0.05)

# The main thread-head orchestration object.
thread_head = ThreadHead(
    0.001,
    [
        hold_pd_controller0,
        hold_pd_controller1
    ],
    {
        'head0': head0,
        'head1': head1
    },
    [] # Utils.
)

# Start the parallel processing.
thread_head.start()

# Helper functions to switch between the different controllers.
def go_pd():
    thread_head.switch_controllers([
        pd_controller0, pd_controller1
    ])

def go_pd_zero():
    init_pos = np.array([-0.15, 0.25, -0.5])
    thread_head.switch_controllers([
        pd_controller0, pd_controller1
    ])

    pd_controller0.update_desired_pos(init_pos)
    pd_controller1.update_desired_pos(init_pos)

def go_pdik():
    thread_head.ti = 0
    thread_head.switch_controllers([ik_ctrl0, ik_ctrl1])

