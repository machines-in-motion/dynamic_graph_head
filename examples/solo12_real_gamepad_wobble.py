import numpy as np
from dynamic_graph_head import ThreadHead, Vicon, HoldPDController
import time
import pinocchio as pin
import os 

import dynamic_graph_manager_cpp_bindings
from robot_properties_solo.solo12wrapper import Solo12Config
from solo12_sim_gamepad_wobble import CentroidalControlWithGamePad

head = dynamic_graph_manager_cpp_bindings.DGMHead(Solo12Config.dgm_yaml_path)

# Create the controllers.
hold_pd_controller = HoldPDController(head, 3., 0.05, with_sliders=True)


log_path = os.path.abspath('')
print("log file path is \n", log_path) 
q0 = Solo12Config.q0.copy()
q0[0] = 0. 
v0 =  Solo12Config.v0.copy()
wobble_controller = CentroidalControlWithGamePad(head, 'solo12/solo12', 
                                    q0, v0, log_path) 

thread_head = ThreadHead(
    0.001,
    hold_pd_controller,
    head,
    [
        ('vicon', Vicon('172.24.117.119:801', ['solo12/solo12']))
    ]
)

# Start the parallel processing.
thread_head.start()



def go_safety():
    thread_head.switch_controllers(thread_head.safety_controllers)



def go_hold_zero():
    go_hold()
    
def go_wobble():
    thread_head.switch_controllers(wobble_controller)

def stop_logging():
    wobble_controller.logger.close_file()
