"""__init__
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import numpy as np

class OCIHead:
    def __init__(self, robot):
        self._robot = robot

    def read(self):
        self._robot.parse_sensor_data()

    def write(self):
        self._robot.send_command()

    def get_sensor(self, sensor_name):
        joints = self._robot.joints
        imu = self._robot.imu

        if sensor_name == 'joint_positions':
            return joints.positions_ref
        elif sensor_name == 'joint_velocities':
            return joints.velocities_ref
        elif sensor_name == 'imu_gyroscope':
            return imu.gyroscope_ref
        elif sensor_name == 'imu_accelerometer':
            return imu.accelerometer_ref
        elif sensor_name == 'imu_attitude_euler':
            return imu.attitude_euler_ref
        elif sensor_name == 'imu_attitude':
            return imu.attitude_quaternion_ref
        elif sensor_name == 'imu_attitude_quaternion':
            return imu.attitude_quaternion_ref
        else:
            raise RuntimeError("Unknown sensor_name '" + sensor_name + "'.")

    def set_control(self, control_name, value):
        joints = self._robot.joints

        if control_name == 'ctrl_joint_torques':
            joints.set_torques(value)
        elif control_name == 'ctrl_joint_positions':
            joints.set_desired_positions(value)
        elif control_name == 'ctrl_joint_velocities':
            joints.set_desired_velocities(value)
        elif control_name == 'ctrl_joint_position_gains':
            joints.set_position_gains(value)
        elif control_name == 'ctrl_joint_velocity_gains':
            joints.set_velocity_gains(value)
        else:
            raise RuntimeError("Unknown control_name '" + control_name + "'.")
