"""
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""
#NOTE: This is only a simulation class for now, we don't have an actual force plate 

from matplotlib.pyplot import hist
import numpy as np
import pinocchio as pin


class SimForcePlate:
    def __init__(self, objects, delay_dt=0, noise_data_std={}):
        """
        Args:
            objects: list of PinBulletWrapper objects

        """ 
        self.objects = objects 
        self.update_delay(delay_dt)
        self.update_noise_data(noise_data_std)

        self.use_delay = True
        self.use_noise_model = True


    def update_delay(self, delay_dt): 
        self.delay_dt = delay_dt
        length = delay_dt + 1

        self.fill_history = True

        # For each object, setup a hisotry for contact forces and active contact status
        self.history = {}
        for i, obj in enumerate(self.objects):
            self.history[obj] = {
                'contact_forces': np.zeros((length, self._robot.nb_ee, 6)),
                'contact_status': np.zeros((length, self._robot.nb_ee))
            } 

    def update_noise_data(self, noise_data_std={}):
        #TODO: Force Plate Noise Not Implemented yet 
        pass 

    def update(self, thread_head):
        history = self.history
        self.write_idx = thread_head.ti % (self.delay_dt + 1)
        self.read_idx = (thread_head.ti + 1) % (self.delay_dt + 1)
        for i, obj in enumerate(self.objects):
            for name, head in thread_head.heads.items():
                if head._robot == obj:
                    history[obj]['contact_forces'][self.write_idx] = head._sensor__force_plate_force.copy()
                    history[obj]['contact_status'][self.write_idx] = head._sensor__force_plate_status.copy()


    def get_contact_force(self, obj): 
        return self.history[obj]['contact_forces'][self.read_idx]

    def get_contact_status(self, obj):
        return self.history[obj]['contact_status'][self.read_idx] 

