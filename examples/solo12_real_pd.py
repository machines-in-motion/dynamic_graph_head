# This a demo to test the ilqr for 2d point mass
import numpy as np

from dynamic_graph_head import ThreadHead, HoldPDController

import time

import dynamic_graph_manager_cpp_bindings
from robot_properties_solo.solo12wrapper import Solo12Config

print("Finished imports")

###
# Create the dgm communication and instantiate the controllers.
head = dynamic_graph_manager_cpp_bindings.DGMHead(Solo12Config.dgm_yaml_path)

head.read()
print(head.get_sensor('slider_positions'))

# Create the controllers.
hold_pd_controller = HoldPDController(head, 3., 0.05, with_sliders=True)

thread_head = ThreadHead(
    0.001, # Run controller at 1000 Hz.
    hold_pd_controller,
    head,
    []
)

# Start the parallel processing.
thread_head.start()

print("Finished controller setup")

input("Wait for input to finish program.")
