import numpy as np

from dynamic_graph_head import ThreadHead, HoldPDController

import time

import dynamic_graph_manager_cpp_bindings
from robot_properties_kuka.config import IiwaConfig
import pinocchio as pin

print("Finished imports")


pin_robot = IiwaConfig.buildRobotWrapper()


class ConstantTorque:
    def __init__(self, head, robot_model, robot_data):
        self.head = head
        
        self.pinModel = robot_model
        self.pinData = robot_data

        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor("joint_velocities")
        self.joint_torques = head.get_sensor("joint_torques_total")
        self.joint_ext_torques = head.get_sensor("joint_torques_external")      
    
    def warmup(self, thread_head):
        pass

    def run(self, thread):

        q = self.joint_positions
        v = self.joint_velocities

        self.tau = np.zeros(7)
        self.tau[1] = 0.0 
        pin.forwardKinematics(self.pinModel, self.pinData, q, v, np.zeros_like(q))
        pin.updateFramePlacements(self.pinModel, self.pinData)
        
        self.g_comp = pin.rnea(self.pinModel, self.pinData, q, v, np.zeros_like(q))
        self.head.set_control('ctrl_joint_torques', self.tau)



###
# Create the dgm communication and instantiate the controllers.
head = dynamic_graph_manager_cpp_bindings.DGMHead(IiwaConfig.yaml_path)


# Create the controllers.
hold_pd_controller = HoldPDController(head, 50., 0.5, with_sliders=False)
tau_ctrl = ConstantTorque(head, pin_robot.model, pin_robot.data)


thread_head = ThreadHead(
    0.001, # Run controller at 1000 Hz.
    hold_pd_controller,
    head,
    []
)

thread_head.switch_controllers(tau_ctrl)

# Start the parallel processing.
thread_head.start()
thread_head.start_logging(5, "test.mds")

print("Finished controller setup")

input("Wait for input to finish program.")
