"""__init__
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import numpy as np

try:
    from vicon_sdk_cpp import ViconClient, ViconFrame
    has_real_vicon = True
except:
    has_real_vicon = False

class SimVicon:
    def __init__(self, objects):
        self.objects = objects
        self.bias_xyz = np.zeros((len(objects), 3))
        self.vicon_frames = {}

    def update(self, thread_head):
        self.vicon_frames = {}

        for i, obj in enumerate(self.objects):
            robot, frame = obj.split('/')
            assert robot == frame, "Not supporting other frames right now."

            # Seek the head with the vicon object.
            for name, head in thread_head.heads.items():
                if head._vicon_name == robot:
                    self.vicon_frames[obj] = {
                        'idx': i,
                        'head': head
                    }


    def get_state(self, vicon_object):
        head = self.vicon_frames[vicon_object]['head']
        pos = head._sensor__vicon_base_position.copy()
        pos[:3] -= self.bias_xyz[self.vicon_frames[vicon_object]['idx']]
        return (
            pos,
            head._sensor__vicon_base_velocity.copy()
        )

    def reset_bias(self, vicon_object):
        self.bias_xyz[:] = 0

    def bias_position(self, vicon_object):
        head = self.vicon_frames[vicon_object]['head']
        self.bias_xyz[self.vicon_frames[vicon_object]['idx'], :2] = head._sensor__vicon_base_position[:2].copy()


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

            self.bias_xy = np.zeros(2)
            self.bias_z = 0.

        def read(self):
            for object, data in self.object_data.items():
                self.client.get_vicon_frame(object, data['frame'])

        def get_state(self, vicon_object):
            data = self.object_data[vicon_object]

            pos = data['frame'].se3_pose.copy()
            pos[:3] -= data['bias']
            return pos, self.frame.velocity_body_frame.copy()

        def bias_position(self, vicon_object):
            data = self.object_data[vicon_object]
            data['bias'][:2] = data['frame'].se3_pose[:2].copy()

        def set_bias_z(self, vicon_object, bias_z):
            self.object_data[vicon_object]['bias'][2] = bias_z
else:
    class Vicon:
        def __init__(self, address, objects):
            raise Exception('vicon_sdk_cpp not found. Is it installed?')



