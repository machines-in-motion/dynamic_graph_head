"""__init__
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import numpy as np

class SimHead:
    def __init__(self, robot, vicon_name='', with_sliders=True):
        self._robot = robot

        self._vicon_name = vicon_name

        # Define the common sensor values.
        nv = robot.pin_robot.model.nv

        # Get number of joints nj
        if robot.useFixedBase:
            nj = nv
        else:
            nj = nv - 6

        self._sensor_joint_positions = np.zeros(nj)
        self._sensor_joint_velocities = np.zeros(nj)

        self.with_sliders = with_sliders
        if self.with_sliders:
            self._sensor_slider_positions = np.zeros(4)

        # If not fixed base, then assume we have an IMU and a vicon.
        if not robot.useFixedBase:
            # Simulated IMU.
            self._sensor_imu_gyroscope = np.zeros(3)

            # Utility for vicon class.
            self._sensor__vicon_base_position = np.zeros(7)
            self._sensor__vicon_base_velocity = np.zeros(6)

        # Controls.
        self._control_ctrl_joint_torques = np.zeros(nj)

    def read(self):
        q, dq = self._robot.get_state()

        self._sensor_joint_positions[:] = q[7:]
        self._sensor_joint_velocities[:] = dq[6:]

        self._sensor_imu_gyroscope[:] = dq[3:6].copy()

        self._sensor__vicon_base_position[:] = q[:7]
        self._sensor__vicon_base_velocity[:] = dq[:6]

        if self.with_sliders:
            for i, l in enumerate(['a', 'b', 'c', 'd']):
                self._sensor_slider_positions[i] = self._robot.get_slider_position(l)

        # TODO: Add noise and delay model.

    def write(self):
        self._robot.send_joint_command(self._control_ctrl_joint_torques)

    def get_sensor(self, sensor_name):
        return self.__dict__['_sensor_' + sensor_name]

    def set_control(self, control_name, value):
        self.__dict__['_control_' + control_name][:] = value

    def reset_state(self, q, dq):
        self._robot.reset_state(q, dq)
