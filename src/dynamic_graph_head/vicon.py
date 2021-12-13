"""
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import numpy as np
import pinocchio as pin

try:
    from vicon_sdk_cpp import ViconClient, ViconFrame
    has_real_vicon = True
except:
    has_real_vicon = False

class SimVicon:
    def __init__(self, objects, delay_dt=0, noise_data_std={}):
        """
        Args:
            noise_model: Std of the noise to apply to the measurements.
        """
        self.objects = objects
        self.bias_xyz = np.zeros((len(objects), 3))
        self.vicon_frames = {}

        self.update_delay(delay_dt)
        self.update_noise_data(noise_data_std)

        self.use_delay = True
        self.use_noise_model = True

    def update_delay(self, delay_dt):
        self.delay_dt = delay_dt
        length = delay_dt + 1

        self.fill_history = True

        # For each object, setup a hisotry for position and body velocity.
        self.history = {}
        for i, obj in enumerate(self.objects):
            self.history[obj] = {
                'position': np.zeros((length, 7)),
                'body_velocity': np.zeros((length, 6))
            }

    def update_noise_data(self, noise_data_std={}):
        self.noise_data_std = noise_data_std
        if not 'position_xyzrpy' in noise_data_std:
            self.noise_data_std['position_xyzrpy'] = np.zeros(6)
        if not 'body_velocity' in noise_data_std:
            self.noise_data_std['body_velocity'] = np.zeros(6)

    def apply_noise_model(self, pos, vel):
        def sample_noise(entry):
            noise_var = self.noise_data_std[entry]**2
            return np.random.multivariate_normal(np.zeros_like(noise_var), np.diag(noise_var))

        noise_pos = sample_noise('position_xyzrpy')

        se3 = pin.XYZQUATToSE3(pos)
        se3.translation += noise_pos[:3]
        se3.rotation = se3.rotation @ pin.rpy.rpyToMatrix(*noise_pos[3:])

        pos = pin.SE3ToXYZQUAT(se3)
        vel += sample_noise('body_velocity')
        return pos, vel

    def update(self, thread_head):
        # Write the position and velocity to the history buffer for each
        # tracked object.
        history = self.history
        self.write_idx = thread_head.ti % (self.delay_dt + 1)
        self.read_idx = (thread_head.ti + 1) % (self.delay_dt + 1)

        for i, obj in enumerate(self.objects):
            robot, frame = obj.split('/')
            assert robot == frame, "Not supporting other frames right now."

            # Seek the head with the vicon object.
            for name, head in thread_head.heads.items():
                if head._vicon_name == robot:
                    pos, vel = self.apply_noise_model(
                        head._sensor__vicon_base_position.copy(),
                        head._sensor__vicon_base_velocity.copy())

                    # At the first timestep, filll the full history.
                    if self.fill_history:
                        self.fill_history = False
                        history[obj]['position'][:] = pos
                        history[obj]['body_velocity'][:] = vel
                    else:
                        history[obj]['position'][self.write_idx] = pos
                        history[obj]['body_velocity'][self.write_idx] = vel

                    self.vicon_frames[obj] = {
                        'idx': i,
                        'head': head,
                    }

    def get_state(self, vicon_object):
        pos = self.history[vicon_object]['position'][self.read_idx]
        vel = self.history[vicon_object]['body_velocity'][self.read_idx]

        pos[:3] -= self.bias_xyz[self.vicon_frames[vicon_object]['idx']]
        return (pos, vel)

    def reset_bias(self, vicon_object):
        self.bias_xyz[self.vicon_frames[vicon_object]['idx']] = 0

    def bias_position(self, vicon_object):
        pos = self.history[vicon_object]['position'][self.read_idx]
        self.bias_xyz[self.vicon_frames[vicon_object]['idx'], :2] = pos[:2].copy()


if has_real_vicon:
    class Vicon:
        """Abstraction around the ViconSDK to read position and velocity"""
        def __init__(self, address, objects):
            self.client = ViconClient()
            self.client.initialize(address)
            self.client.run()

            self.objects = objects

            self.object_data = {}
            for i, object in enumerate(objects):
                self.object_data[object] = {
                    'frame': ViconFrame(),
                    'bias': np.zeros(3)
                }

                print('Vicon: Tracking object', object)

        def update(self, thread_head):
            for object, data in self.object_data.items():
                self.client.get_vicon_frame(object, data['frame'])

        def get_state(self, vicon_object):
            data = self.object_data[vicon_object]

            pos = data['frame'].se3_pose.copy()
            pos[:3] -= data['bias']
            return pos, data['frame'].velocity_body_frame.copy()

        def bias_position(self, vicon_object):
            data = self.object_data[vicon_object]
            data['bias'][:2] = data['frame'].se3_pose[:2].copy()

        def set_bias_z(self, vicon_object, bias_z):
            self.object_data[vicon_object]['bias'][2] = bias_z
else:
    class Vicon:
        def __init__(self, address, objects):
            raise Exception('vicon_sdk_cpp not found. Is it installed?')



