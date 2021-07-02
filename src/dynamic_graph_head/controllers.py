"""__init__
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import numpy as np

class ZeroTorquesController:
    def __init__(self, head):
        # Zero the commands.
        self.tau = np.zeros_like(head.get_sensor('joint_positions'))

    def warmup(self, thread_head):
        pass

    def run(self, thread_head):
        thread_head.head.set_control('ctrl_joint_torques', self.tau)


class HoldPDController:
    def __init__(self, head, Kp, Kd, with_sliders=False):
        self.head = head
        self.Kp = Kp
        self.Kd = Kd

        self.slider_scale = np.pi
        self.with_sliders = with_sliders

        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')

        if with_sliders:
            self.slider_positions = head.get_sensor('slider_positions')

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
            for i in range(4):
                sliders_out[3 * i + 0] = slider_A
                sliders_out[3 * i + 1] = slider_A
                sliders_out[3 * i + 2] = 2. * (1. - slider_A)

                if i >= 2:
                    sliders_out[3 * i + 1] *= -1
                    sliders_out[3 * i + 2] *= -1

            # Swap the hip direction.
            sliders_out[3] *= -1
            sliders_out[9] *= -1

        elif self.joint_positions.shape[0] == 8:  #  solo8
            slider_A = sliders[0]
            for i in range(4):
                sliders_out[2 * i + 0] = slider_A
                sliders_out[2 * i + 1] = 2. * (1. - slider_A)

                if i >= 2:
                    sliders_out[2 * i + 0] *= -1
                    sliders_out[2 * i + 1] *= -1
        return sliders_out

    def run(self, thread_head):
        if self.with_sliders:
            self.des_position = self.slider_scale * (
                self.map_sliders(self.slider_positions) - self.slider_zero_pos) + self.zero_pos
        else:
            self.des_position = self.zero_pos

        self.tau = self.Kp * (self.des_position - self.joint_positions) - self.Kd * self.joint_velocities
        thread_head.head.set_control('ctrl_joint_torques', self.tau)
