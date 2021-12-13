"""__init__
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import numpy as np

class SimHead:
    def __init__(self, robot, vicon_name='', with_sliders=True, joint_index=None,
                 measurement_delay_dt=0, control_delay_dt=0, noise_data_std={}):
        self._robot = robot
        self._vicon_name = vicon_name
        self._joint_index = joint_index

        # Define the common sensor values.
        nv = robot.pin_robot.model.nv

        # Get number of joints nj
        if robot.useFixedBase:
            if joint_index is None:
                nj = nv
            else:
                nj = len(joint_index)
        else:
            nj = nv - 6


        self.nj = nj
        self._sensor_joint_positions = np.zeros(nj)
        self._sensor_joint_velocities = np.zeros(nj)

        self.with_sliders = with_sliders
        if self.with_sliders:
            self._sensor_slider_positions = np.zeros(4)

        # If not fixed base, then assume we have an IMU and a vicon.
        if not robot.useFixedBase:
            # Simulated IMU.
            self._sensor_imu_gyroscope = np.zeros(3)
            self._sensor_imu_accelerometer = np.zeros(3)
            # Utility for vicon class.
            self._sensor__vicon_base_position = np.zeros(7)
            self._sensor__vicon_base_velocity = np.zeros(6)
            # Utility for force plate. 
            self._sensor__force_plate_force = np.zeros((self._robot.nb_ee, 6))
            self._sensor__force_plate_status = np.zeros(self._robot.nb_ee)
        # Controls.
        self._control_ctrl_joint_torques = np.zeros(nj)

        self.update_noise_data(noise_data_std)
        self.update_control_delay(control_delay_dt)
        self.update_measurement_delay(measurement_delay_dt)

    def update_noise_data(self, noise_data_std):
        self._noise_data_std = noise_data_std
        if not 'joint_positions' in noise_data_std:
            self._noise_data_std['joint_positions'] = np.zeros(self.nj)
        if not 'joint_velocities' in noise_data_std:
            self._noise_data_std['base_velocity'] = np.zeros(self.nj)
        if not 'imu_gyroscope' in noise_data_std:
            self._noise_data_std['imu_gyroscope'] = np.zeros(3)
        if not 'imu_accelerometer' in noise_data_std:
            self._noise_data_std['imu_accelerometer'] = np.zeros(3)


    def update_control_delay(self, delay_dt):
        self._fill_history_control = True
        self._ti = 0
        self._control_delay_dt = delay_dt

        length = delay_dt + 1

        self._history_control = {
            'ctrl_joint_torques': np.zeros((length, self.nj))
        }

    def update_measurement_delay(self, delay_dt):
        self._fill_history_measurement = True
        self._ti = 0
        self._measurement_delay_dt = delay_dt

        length = delay_dt + 1

        self._history_measurements = {
            'joint_positions': np.zeros((length, self.nj)),
            'joint_velocities': np.zeros((length, self.nj)),
            'imu_accelerometer': np.zeros((length, 3)),
            'imu_gyroscope': np.zeros((length, 3)), 
        }

    def sample_noise(self, entry):
        noise_var = self._noise_data_std[entry]**2
        return np.random.multivariate_normal(np.zeros_like(noise_var), np.diag(noise_var))

    def read(self):
        q, dq = self._robot.get_state()

        write_idx = self._ti % (self._measurement_delay_dt + 1)
        if self._fill_history_measurement:
            self._fill_history_measurement = False
            write_idx = None
        read_idx = (self._ti + 1) % (self._measurement_delay_dt + 1)

        history = self._history_measurements

        if not self._robot.useFixedBase:
            # Write to the measurement history with noise.
            history['joint_positions'][write_idx] = q[7:]
            history['joint_velocities'][write_idx] = dq[6:]
            history['imu_gyroscope'][write_idx] = self._robot.get_base_imu_angvel()
            history['imu_accelerometer'][write_idx] = self._robot.get_base_imu_linacc() 
            self._sensor_imu_gyroscope[:] = history['imu_gyroscope'][read_idx]
            self._sensor_imu_accelerometer[:] = history['imu_accelerometer'][read_idx]
            self._sensor__vicon_base_position[:] = q[:7]
            self._sensor__vicon_base_velocity[:] = dq[:6]
            # only read forces for free floating for now 
            contact_status, contact_forces = self._robot.get_force()
            for i, cnt_id in enumerate(self._robot.pinocchio_endeff_ids):
                self._sensor__force_plate_force[i,:] = contact_forces[i][:]
                self._sensor__force_plate_status[i] = contact_status[i] 
        else:
            if self._joint_index:
                history['joint_positions'][write_idx] = q[self._joint_index]
                history['joint_velocities'][write_idx] = dq[self._joint_index]
            else:
                history['joint_positions'][write_idx] = q
                history['joint_velocities'][write_idx] = dq

        self._sensor_joint_positions[:] = history['joint_positions'][read_idx]
        self._sensor_joint_velocities[:] = history['joint_velocities'][read_idx]

        if self.with_sliders:
            for i, l in enumerate(['a', 'b', 'c', 'd']):
                self._sensor_slider_positions[i] = self._robot.get_slider_position(l)

    def write(self):
        write_idx = self._ti % (self._measurement_delay_dt + 1)
        if self._fill_history_control:
            self._fill_history_control = False
            write_idx = None
        read_idx = (self._ti + 1) % (self._measurement_delay_dt + 1)

        history = self._history_control
        history['ctrl_joint_torques'][write_idx] = self._control_ctrl_joint_torques
        
        self._last_ctrl_joint_torques = history['ctrl_joint_torques'][read_idx]
        self._ti += 1

    def sim_step(self):
        self._robot.send_joint_command(self._last_ctrl_joint_torques)

    def get_sensor(self, sensor_name):
        return self.__dict__['_sensor_' + sensor_name]

    def set_control(self, control_name, value):
        self.__dict__['_control_' + control_name][:] = value

    def reset_state(self, q, dq):
        self._robot.reset_state(q, dq)
