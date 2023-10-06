import pinocchio
import numpy as np
from pathlib import Path

import dynamic_graph_manager_cpp_bindings
from dynamic_graph_head import ThreadHead, Vicon, SimHead, SimVicon, HoldPDController, StateEstimator

from mim_control_cpp import CentroidalImpedanceController

from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from bullet_utils.env import BulletEnvWithGround


class EstimatorCentroidalController:
    def __init__(self, head, vicon_name, mu, kp, kd, kc, dc, kb, db):
        self.set_k(kp, kd)
        self.robot = Solo12Config.buildRobotWrapper()
        self.robot_config = Solo12Config()
        self.vicon_name = vicon_name

        qp_penalty_weights = np.array([5e5, 5e5, 5e5, 1e6, 1e6, 1e6])

        q_init = self.robot_config.q0.copy()
        q_init[0] = 0.

        self.ctrl = CentroidalImpedanceController()
        self.ctrl.initialize(
            2.5,
            np.diag(self.robot.mass(q_init)[3:6, 3:6]),
            self.robot.model,
            "universe",
            self.robot_config.end_effector_names,
            mu,
            qp_penalty_weights,
            kc, dc, kb, db,
            self.kp, self.kd
        )

        # Desired center of mass position and velocity.
        self.x_com = [0.0, 0.0, 0.225]
        self.xd_com = [0.0, 0.0, 0.0]
        # The base should be flat.
        self.x_ori = [0.0, 0.0, 0.0, 1.0]
        self.x_angvel = [0.0, 0.0, 0.0]

        # Desired leg length.
        self.x_des = [
             0.195,  0.147, 0.015, 0, 0, 0, 1.,
             0.195, -0.147, 0.015, 0, 0, 0, 1.,
            -0.195,  0.147, 0.015, 0, 0, 0, 1.,
            -0.195, -0.147, 0.015, 0, 0, 0, 1.
        ]
        self.xd_des = np.zeros(4 * 6)

        self.tau = np.zeros(12)
        self.head = head
        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')
        self.imu_gyroscope = head.get_sensor('imu_gyroscope')

        # Feedback condition, True; using Vicon states for feedback, False; Estimator states.
        self.use_vicon = True
        self.ti = 0

    def set_k(self, kp, kd):
        self.kp = [kp, kp, kp, 0, 0, 0]
        self.kd = [kd, kd, kd, 0, 0, 0]

    def warmup(self, thread):
        self.ti = 0
        thread.vicon.bias_position(self.vicon_name)
        thread.estimator.set_settings([0.66, 0.17])
    
    def set_vicon(self, bool):
        self.use_vicon = bool
    
    def get_base_vicon(self, thread):
        base_pos, base_vel = thread.vicon.get_state(self.vicon_name)
        base_vel[3:] = self.imu_gyroscope
        return base_pos, base_vel

    def get_base_estimator(self, thread_head):
        return (
            thread_head.estimator.estimator_base_position,
            thread_head.estimator.estimator_base_velocity
        )
    
    def get_estimator_force(self, thread_head):
        return thread_head.estimator.estimator_ee_forces_norm

    def set_tau(self, thread_head):
        thread_head.estimator.set_joint_torques(self.tau)

    def get_target_configuration(self, ti):
        return (
            np.array([1. ,1. , 1., 1.]), self.x_com, self.xd_com, 
            self.x_ori, self.x_angvel, self.x_des, self.xd_des
        )

    def run(self, thread):
        base_pos_vicon, base_vel_vicon = self.get_base_vicon(thread)
        base_pos_estimator, base_vel_estimator = self.get_base_estimator(thread)

        # Store the vicon and estimator on `self` for logging.
        self.vicon_base_pos = base_pos_vicon
        self.vicon_base_vel = base_vel_vicon
        self.estimator_base_pos = base_pos_estimator
        self.estimator_base_vel = base_vel_estimator
        self.estimator_ee_forces = self.get_estimator_force(thread)
        
        if self.use_vicon:
            base_pos = base_pos_vicon
            base_vel = base_vel_vicon
        else:
            base_pos = base_pos_estimator
            # Get the Z-position from Vicon; EKF Z-position is diverging.
            base_pos[2] = base_pos_vicon[2]
            base_vel = base_vel_estimator

        self.q = np.hstack([base_pos, self.joint_positions])
        self.dq = np.hstack([base_vel, self.joint_velocities])

        quat = pinocchio.Quaternion(self.q[6], self.q[3], self.q[4], self.q[5])

        cnt_array, x_com, xd_com, x_ori, x_angvel, x_des, xd_des = \
            self.get_target_configuration(self.ti)
        
        self.ctrl.run(
            self.q, self.dq,
            cnt_array,
            self.q[:3],
            x_com,
            quat.toRotationMatrix().dot(self.dq[:3]), # local to world frame
            xd_com,
            self.q[3:7],
            x_ori,
            self.dq[3:6],
            x_angvel,
            x_des,
            xd_des
        )

        self.tau[:] = self.ctrl.get_joint_torques()
        self.head.set_control('ctrl_joint_torques', self.tau)
        self.set_tau(thread)

        self.ti += 1


class EstimatorCentroidalControllerReplay(EstimatorCentroidalController):
    def __init__(self, head, vicon_name, mu, kp, kd, kc, dc, kb, db, filepath):
        super().__init__(head, vicon_name, mu, kp, kd, kc, dc, kb, db)

        self.trajectory_data = np.load(filepath)
        self.traj_contact_activation = self.trajectory_data['contact_activation']
        self.traj_com_pos = self.trajectory_data['com_position']
        self.traj_com_vel = self.trajectory_data['com_velocity']
        self.traj_base_ori = self.trajectory_data['base_orientation']
        self.traj_base_ang_vel = self.trajectory_data['base_angular_velocity']
        self.traj_ee_placement = self.trajectory_data['end_frame_placement']
        self.traj_ee_vel = self.trajectory_data['end_frame_velocity']

        self.get_target_configuration(self.ti)

    def get_target_configuration(self, ti):
        return (
            self.traj_contact_activation[ti],
            self.traj_com_pos[ti],
            self.traj_com_vel[ti],
            self.traj_base_ori[ti],
            self.traj_base_ang_vel[ti],
            self.traj_ee_placement[ti],
            self.traj_ee_vel[ti],
        )

    def warmup(self, thread):
        self.ti = 0
        thread.vicon.bias_position(self.vicon_name)
        self.set_vicon(True)

        # Modify the force thresholds for different motions here.
        thread.estimator.set_settings(force_thresholds=[1.6, 1.45])

def run_simulation():
    # Create a Pybullet simulation environment.
    bullet_env = BulletEnvWithGround()

    # Create a robot instance. This initializes the simulator as well.
    robot = Solo12Robot()
    bullet_env.add_robot(robot)

    # Create the dgm communication.
    head = SimHead(robot, vicon_name='solo12')

    # Create the controllers
    hold_pd_controller = HoldPDController(head, 3.0, 0.05, with_sliders=False)

    centroidal_ctrl = EstimatorCentroidalController(head, 'solo12/solo12', 0.6, 50.0, 0.9,
    [200., 200., 200.], [15., 15., 15.], [25., 25., 25.], [22.5, 22.5, 22.5],
    )

    # Include the path for the desired trajectory file here
    file_path = Path.home() / "data" / "trajectory" / "solo12_trot.npz"
    centroidal_ctrl_replay = EstimatorCentroidalControllerReplay(head, 'solo12/solo12', 0.6, 50.0, 0.9,
    [200., 200., 200.], [15., 15., 15.], [25., 25., 25.], [22.5, 22.5, 22.5], file_path
    )

    # Setup the simulated head and the thread_head.
    thread_head = ThreadHead(
        0.001,
        hold_pd_controller,
        head,
        [
            ('vicon', SimVicon(['solo12/solo12'])),
            ('estimator', StateEstimator(np.array([1e-5, 1e-6, 1e-7]), head))
        ],
        bullet_env # Environment to step.
    )

    q0 = np.array(Solo12Config.initial_configuration)
    q0[0] = 0.
    q0[2] -= 0.01
    dq0 = np.array(Solo12Config.initial_velocity)
    print("Initial_Configuration",q0)
    print("Initial_Velocity", dq0)

    print("\nRunning the Simulation")
    head.reset_state(q0, dq0)
    thread_head.sim_run(1000)

    print("\nSwitching to Centroidal_Controller")
    thread_head.switch_controllers(centroidal_ctrl)
    thread_head.sim_run(1000)

    print("\nSwitching to Centroidal_Controller_Replay")
    thread_head.switch_controllers(centroidal_ctrl_replay)
    thread_head.start_streaming()
    thread_head.start_logging()
    thread_head.sim_run(10000)
    thread_head.stop_logging()
    thread_head.stop_streaming()


    print("######### END #########")


if __name__ == "__main__":
    # Run the simulation.
    # run_simulation()

    # Run on the real robot.
    # Create the dgm communication.
    head = dynamic_graph_manager_cpp_bindings.DGMHead(Solo12Config.dgm_yaml_path)
    
    # Create the controllers
    hold_pd_controller = HoldPDController(head, 3.0, 0.05, with_sliders=True)

    centroidal_ctrl = EstimatorCentroidalController(head, 'solo12/solo12', 0.2, 50., 0.9,
    [200., 200., 200.], [25., 15., 15.], [20., 30., 22.], [4, 10, 10],
    )

    # Include the path for the desired trajectory file here
    file_path = Path.home() / "data" / "trajectory" / "solo12_trot.npz"
    centroidal_ctrl_replay = EstimatorCentroidalControllerReplay(head, 'solo12/solo12', 0.2, 50., 0.9,
    [200., 200., 200.], [25., 15., 15.], [20., 30., 22.], [4, 10, 10], file_path,
    )
    thread_head = ThreadHead(
        0.001,
        hold_pd_controller,
        head,
        [
            ('vicon', Vicon('10.32.27.53:801', ['solo12/solo12'])),
            ('estimator', StateEstimator(np.array([1e-5, 1e-6, 1e-7]), head))
        ]   
    )
    thread_head.start()

###
# List of helper go functions.
def go_hold():
    thread_head.switch_controllers(thread_head.safety_controllers)

def go_zero():
    go_hold()
    hold_pd_controller.go_zero()

def go_cent():
    thread_head.switch_controllers(centroidal_ctrl)

def follow_traj():
    thread_head.switch_controllers(centroidal_ctrl_replay)
