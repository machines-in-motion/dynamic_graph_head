"""Basic Solo12 example using sliders and the on-board PD controller."""
from dynamic_graph_head import ThreadHead
from dynamic_graph_head.controllers import HoldOnBoardPDController
import dynamic_graph_manager_cpp_bindings
from robot_properties_solo.solo12wrapper import Solo12Config

if __name__ == "__main__":

    ###
    # Create the dgm communication and instantiate the controllers.
    head = dynamic_graph_manager_cpp_bindings.DGMHead(Solo12Config.dgm_yaml_path)

    head.read()
    print(head.get_sensor("slider_positions"))

    # Create the controllers.
    hold_pd_controller = HoldOnBoardPDController(head, 3.0, 0.05, with_sliders=True)

    thread_head = ThreadHead(
        0.001, hold_pd_controller, head, []  # Run controller at 1000 Hz.
    )

    # Start the parallel processing.
    thread_head.start()

    print("Finished controller setup")

    input("Wait for input to finish program.")
