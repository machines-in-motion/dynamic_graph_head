"""__init__
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""
import abc

import numpy as np


class ZeroTorquesController:
    def __init__(self, head):
        # Zero the commands.
        self.tau = np.zeros_like(head.get_sensor("joint_positions"))

    def warmup(self, thread_head):
        pass

    def run(self, thread_head):
        thread_head.head.set_control("ctrl_joint_torques", self.tau)


class BaseHoldPDController(abc.ABC):
    """Base for example controllers using position control.

    If `with_sliders` is set, hardware sliders are used to set the target
    position.  Otherwise the target position is fixed to zero for all joints.
    """

    def __init__(self, head, Kp: float, Kd: float, with_sliders: bool=False):
        """
        Args:
            head: Instance of DGHead or SimHead.
            Kp: P-gain for the PD controller.  The same gain is used for all
                joints.
            Kd: D-gain for the PD controller.  The same gain is used for all
                joints.
            with_sliders: If true, use hardware sliders to get target position.
                Otherwise target is fixed to zero.
        """
        self.head = head
        self.Kp = Kp
        self.Kd = Kd

        self.slider_scale = np.pi
        self.with_sliders = with_sliders

        self.joint_positions = head.get_sensor("joint_positions")
        self.joint_velocities = head.get_sensor("joint_velocities")

        if with_sliders:
            self.slider_positions = head.get_sensor("slider_positions")

    def warmup(self, thread_head):
        self.zero_pos = self.joint_positions.copy()

        if self.with_sliders:
            self.slider_zero_pos = self.map_sliders(self.slider_positions)

    def go_zero(self):
        # TODO: Make this an interpolation.
        self.zero_pos = np.zeros_like(self.zero_pos)

        if self.with_sliders:
            self.slider_zero_pos = self.map_sliders(self.slider_positions)

    def map_sliders(self, sliders):
        sliders_out = np.zeros_like(self.joint_positions)

        if self.joint_positions.shape[0] == 12:  # solo12
            slider_A = sliders[0]
            slider_B = sliders[1]
            for i in range(4):
                sliders_out[3 * i + 0] = slider_A
                sliders_out[3 * i + 1] = slider_B
                sliders_out[3 * i + 2] = 2.0 * (1.0 - slider_B)

                if i >= 2:
                    sliders_out[3 * i + 1] *= -1
                    sliders_out[3 * i + 2] *= -1

            # Swap the hip direction.
            sliders_out[3] *= -1
            sliders_out[9] *= -1

        elif self.joint_positions.shape[0] == 8:  # solo8
            slider_A = sliders[0]
            for i in range(4):
                sliders_out[2 * i + 0] = slider_A
                sliders_out[2 * i + 1] = 2.0 * (1.0 - slider_A)

                if i >= 2:
                    sliders_out[2 * i + 0] *= -1
                    sliders_out[2 * i + 1] *= -1

        elif self.joint_positions.shape[0] == 6:  # bolt
            slider_A = sliders[0]
            slider_B = sliders[1]

            for i in range(2):
                sliders_out[3 * i + 0] = slider_A
                sliders_out[3 * i + 1] = slider_B
                sliders_out[3 * i + 2] = 1.0 - slider_B

            sliders_out[3] *= -1

        elif self.joint_positions.shape[0] == 2:  # teststand
            slider_A = sliders[0]
            sliders_out[0] = slider_A
            sliders_out[1] = 2.0 * (1.0 - slider_A)

        return sliders_out

    def get_desired_position(self):
        if self.with_sliders:
            des_position = (
                self.slider_scale
                * (self.map_sliders(self.slider_positions) - self.slider_zero_pos)
                + self.zero_pos
            )
        else:
            des_position = self.zero_pos

        return des_position

    @abc.abstractmethod
    def run(self, thread_head):
        pass


class HoldPDController(BaseHoldPDController):
    """Example controller using PD control and sending torque commands.

    Runs a PD controller and sends the corresponding torque commands to the
    robot.  Works for robots that provide a command "ctrl_joint_torques".
    """

    def run(self, thread_head):
        self.des_position = self.get_desired_position()

        self.tau = (
            self.Kp * (self.des_position - self.joint_positions)
            - self.Kd * self.joint_velocities
        )
        self.head.set_control("ctrl_joint_torques", self.tau)


class HoldOnBoardPDController(BaseHoldPDController):
    """Example controller using the on-board PD controller.

    Uses the on-board PD controller of the master board to control the position
    of the joints.  Works for robots providing commands "ctrl_joint_positions",
    "ctrl_joint_velocities", "ctrl_joint_position_gains",
    "ctrl_joint_velocity_gains".
    """

    def __init__(self, head, Kp: float, Kd: float, with_sliders: bool = False):
        super().__init__(head, Kp, Kd, with_sliders)

        # set gains of the on-board controller
        kp_array = np.full_like(self.joint_positions, Kp)
        kd_array = np.full_like(self.joint_positions, Kd)
        self.head.set_control("ctrl_joint_position_gains", kp_array)
        self.head.set_control("ctrl_joint_velocity_gains", kd_array)

        self._sensor_msgs_last_print = 0

    def run(self, thread_head):
        self.des_position = self.get_desired_position()
        des_velocity = np.zeros_like(self.des_position)

        self.head.set_control("ctrl_joint_positions", self.des_position)
        self.head.set_control("ctrl_joint_velocities", des_velocity)

        sensor_msg_sent = self.head.get_sensor("sent_sensor_messages")
        sensor_msg_lost = self.head.get_sensor("lost_sensor_messages")

        if sensor_msg_sent - self._sensor_msgs_last_print > 1000:
            print(
                "Lost {}/{} sensor messages".format(
                    sensor_msg_lost, sensor_msg_sent
                )
            )
            self._sensor_msgs_last_print = sensor_msg_sent
