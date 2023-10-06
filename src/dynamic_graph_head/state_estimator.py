import pinocchio as pin
import numpy as np

from mim_estimation_cpp import RobotStateEstimator, RobotStateEstimatorSettings
from robot_properties_solo.solo12wrapper import  Solo12Config


class StateEstimator:
    def __init__(self, measurement_cov, head, force_thresholds=[0.66, 0.17]):
        self.robot_config = Solo12Config()

        # Create the Estimator setting instance.
        self.estimator_settings = RobotStateEstimatorSettings()
        self.estimator_settings.pinocchio_model = self.robot_config.pin_robot.model

        self.estimator_settings.imu_in_base = pin.SE3(
            self.robot_config.rot_base_to_imu.T,
            self.robot_config.r_base_to_imu,
        )
        self.estimator_settings.is_imu_frame = False
        self.estimator_settings.end_effector_frame_names = (
            self.robot_config.end_effector_names
        )
        self.estimator_settings.urdf_path = self.robot_config.urdf_path

        self.robot_weight_per_ee = self.robot_config.mass * 9.81 / 4

        # Create the estimator and initialize it.
        self.estimator = RobotStateEstimator()

        self.estimator_settings.force_threshold_up = (
            max(force_thresholds) * self.robot_weight_per_ee
        )
        self.estimator_settings.force_threshold_down = (
            min(force_thresholds) * self.robot_weight_per_ee
        )

        self.estimator_settings.meas_noise_cov = measurement_cov

        self.initialize_filter()

        # Get the reference to the sensors.
        self.head = head
        self.imu_accelerometer = head.get_sensor("imu_accelerometer")
        self.imu_gyroscope = head.get_sensor("imu_gyroscope")
        self.joint_positions = head.get_sensor("joint_positions")
        self.joint_velocities = head.get_sensor("joint_velocities")
        self.joint_torques = np.zeros(12)

    def initialize_filter(self):
        # Initialize the estimator.
        self.estimator.initialize(self.estimator_settings)

        # Set the initial values of the estimator.
        self.estimator_init_pos = np.array(self.robot_config.initial_configuration)
        self.estimator_init_pos[0] = self.estimator_init_pos[1] = 0.0
        self.estimator_init_pos[2] = 0.24
        self.estimator_init_vel = np.array(self.robot_config.initial_velocity)
        self.estimator.set_initial_state(
            self.estimator_init_pos, self.estimator_init_vel
            )
    
    def set_joint_torques(self, tau):
        self.joint_torques = tau

    def set_settings(self, force_thresholds):
        self.estimator_settings.force_threshold_up = (
            max(force_thresholds) * self.robot_weight_per_ee
        )
        self.estimator_settings.force_threshold_down = (
            min(force_thresholds) * self.robot_weight_per_ee
        )
        self.estimator.set_settings(self.estimator_settings)

    def update(self, thread_head):
        # Run the estimator.
        self.estimator.run(
            self.imu_accelerometer, self.imu_gyroscope,
            self.joint_positions, self.joint_velocities,
            self.joint_torques)

        # Store values for others to read.
        self.estimator_base_position = np.zeros(7)
        self.estimator_base_velocity = np.zeros(6)
        self.estimator_ee_forces_norm = np.zeros(6)
        self.estimator.get_state(self.estimator_base_position, self.estimator_base_velocity)
        forces = [
            self.estimator.get_force(ee)
            for ee in self.estimator_settings.end_effector_frame_names
        ]
        self.estimator_ee_forces_norm[0:4] = np.array([np.linalg.norm(f) for f in forces])
        
        # The last two elements are the upper and lower thresholds on forces.
        self.estimator_ee_forces_norm[4:6] = np.array([self.estimator_settings.force_threshold_up, 
                                                    self.estimator_settings.force_threshold_down])
