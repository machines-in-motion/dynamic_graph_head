import numpy as np
from dynamic_graph_head import ThreadHead, HoldPDController
import time
from robot_properties_bolt.bolt_wrapper import BoltRobot, BoltConfig
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hardware',help='flag to activate hardware test',default=False, action='store_true')
    args = parser.parse_args() 

    if args.hardware:

        import dynamic_graph_manager_cpp_bindings
        # Create the dgm communication and instantiate the controllers.
        head = dynamic_graph_manager_cpp_bindings.DGMHead(BoltConfig.dgm_yaml_path)

        head.read()
        print(head.get_sensor('slider_positions'))

    else:

        import pybullet
        from bullet_utils.env import BulletEnvWithGround
        from dynamic_graph_head import SimHead, SimVicon

        bullet_env = BulletEnvWithGround()

        # Create a robot instance. This initializes the simulator as well.
        # robot = BoltRobot(use_fixed_base=True)
        robot = BoltRobot()

        bullet_env.add_robot(robot)
        q0 = np.array(BoltConfig.initial_configuration)
        q0[0] = 0.
        q0[2] -= 0.01

        dq0 = np.array(BoltConfig.initial_velocity)

        # explicitly creating constraint to keep bolt in air
        plane_id = bullet_env.objects[0]
        pybullet.createConstraint(
                                plane_id,
                                -1,
                                robot.robotId,
                                -1,
                                jointType=pybullet.JOINT_FIXED,
                                jointAxis=[0,0,1],
                                parentFramePosition= [0,0,0.65],
                                childFramePosition=[0,0,0]
                                )


        # Create the controllers.
        head = SimHead(robot, vicon_name='bolt')
        head.reset_state(q0, dq0)
    
    # common for both real hardware and simulation
    thread_head = ThreadHead(
        0.001, # dt.
        HoldPDController(head, 3., 0.05, True), # Safety controllers.
        head, # Heads to read / write from.
        [     # Utils.
            ('vicon', SimVicon(['bolt/bolt']))
        ], 
        bullet_env # Environment to step.
    )


    # Start the parallel processing.
    thread_head.start()

    print("Finished controller setup")
    input("Wait for input to finish program.")
