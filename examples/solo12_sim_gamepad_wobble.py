""" a demo with solo 12 taking commands from joystick and running centroidal controller to move in simulation 
an EKF is added on top to estimate the contact and base state, the motion will be restricted to base wobbling 
for the controller this means that end effector commands will never change 
i.e. always active contacts and feet at the same initial configuration 

LeftJoystickY: pitch 
LeftJoystickX: roll 
RightJoystickX : yaw

The com reference will remain fixed around that of the original q0 from Solo12Config

 """ 


import numpy as np
import os, time 
from copy import deepcopy
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
import pinocchio as pin
from mim_estimation_cpp import RobotStateEstimator, RobotStateEstimatorSettings
from mim_data_utils import DataLogger
from bullet_utils.env import BulletEnvWithGround
import pybullet as p 
from dynamic_graph_head import ThreadHead, SimHead, SimVicon, HoldPDController, SimForcePlate
from inputs import get_gamepad 

from mim_control_cpp import CentroidalImpedanceController 
import threading 
import math 
#________ GamePad Thread ________#
gamepad_values = np.zeros(3)

MAX_TRIG_VAL = math.pow(2, 8)
MAX_JOY_VAL = math.pow(2, 15)

def gamepad_thread_fn():
    global gamepad_values

    try:
        get_gamepad()
    except:
        print('!!!    NO GAMEPAD FOUND       !!!')
        return

    while (True):
        events = get_gamepad()
        for event in events:
            if event.ev_type == "Absolute":
                if event.code == 'ABS_Y':
                    gamepad_values[1] = .5 * event.state / MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    gamepad_values[0] = .5 * event.state / MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    gamepad_values[2] = .5 * event.state / MAX_JOY_VAL # normalize between -1 and 1



class CentroidalControlWithGamePad:
    def __init__(self, head, vicon_name, q0, v0, log_path=None, sim_forces=False):
        #________ PD Parameters ________#
        self.head = head
        self.scale = np.pi
        self.q0 = q0 
        self.q0[2] -= .012
        self.v0 = v0
        self.d = 0. 
        self.t = 0 
        self.sim_forces = False # activates force readings from simulation

        #________ Robot Parameters ________#
        self.robot_config = Solo12Config() 
        self.pin_robot = self.robot_config.buildRobotWrapper()
        self.vicon_name = vicon_name
        self.contact_names = self.robot_config.end_effector_names


        #________ Reference CoM & Feet Positions ________#
        self.com_ref = pin.centerOfMass(self.pin_robot.model, self.pin_robot.data, self.q0)
        pin.framesForwardKinematics(self.pin_robot.model, self.pin_robot.data, self.q0)
        self.contact_positions = np.zeros(len(self.contact_names)*7)
        for i, name in enumerate(self.contact_names):
            self.contact_positions[7*i:7*i+3] = self.pin_robot.data.oMf[self.pin_robot.model.getFrameId(name)].translation 
            self.contact_positions[7*i+6] = 1.

        #________ Data Logs ________#
        self.tau = np.zeros(self.pin_robot.nv-6)
        self.q_sim = np.zeros(self.pin_robot.nq)
        self.v_sim = np.zeros(self.pin_robot.nv)
        self.q_est = np.zeros(self.pin_robot.nq)
        self.v_est = np.zeros(self.pin_robot.nv)
        self.x_sim = np.zeros(self.pin_robot.nq+ self.pin_robot.nv)
        self.x_est = np.zeros(self.pin_robot.nq+ self.pin_robot.nv)
        self.f_sim = np.zeros([self.robot_config.nb_ee,6])
        self.f_est = np.zeros([self.robot_config.nb_ee,6])
        self.c_sim = np.zeros(self.robot_config.nb_ee)
        self.c_est = np.zeros(self.robot_config.nb_ee)    
        self.u_applied = np.zeros(self.pin_robot.nv-6)
        self.u_observed = np.zeros(self.pin_robot.nv-6)
        self.imu_linacc_sim = np.zeros(3)
        self.imu_angvel_sim = np.zeros(3) 
        self.imu_linacc_est = np.zeros(3)
        self.imu_angvel_est = np.zeros(3) 
    
        #________ initialze impendace controller ________#
        mu = 0.2
        kc = np.array([100., 100., 100.]) 
        dc = np.array( [15., 15., 15.]) 
        kb = np.array([25, 25, 25]) 
        db = np.array([22.5, 22.5, 22.5]) 
        qp_penalty_weights = np.array([5e5, 5e5, 5e5, 1e6, 1e6, 1e6])
        # impedance gains
        kp = np.array([50, 50, 50, 0, 0, 0])
        kd = np.array([.7, .7, .7, 0, 0, 0])        

        self.controller = CentroidalImpedanceController()
        self.controller.initialize(
            2.5,
            np.diag(self.pin_robot.mass(self.q0)[3:6, 3:6]),
            self.pin_robot.model,
            "universe",
            self.robot_config.end_effector_names,
            mu,
            qp_penalty_weights,
            kc, dc, kb, db,
            kp, kd
        )

        #________ initialze estimator ________#
        estimator_settings = RobotStateEstimatorSettings()
        estimator_settings.is_imu_frame = False
        estimator_settings.pinocchio_model = self.pin_robot.model

        # IMU pose offset in base frame
        self.rot_base_to_imu = np.identity(3)
        self.r_base_to_imu = np.array([0.10407, -0.00635, 0.01540])

        estimator_settings.imu_in_base = pin.SE3(self.rot_base_to_imu.T, self.r_base_to_imu)
        estimator_settings.end_effector_frame_names = (self.robot_config.end_effector_names)
        estimator_settings.urdf_path = self.robot_config.urdf_path
        robot_weight_per_ee = self.robot_config.mass * 9.81 / 4
        estimator_settings.force_threshold_up = 0.8 * robot_weight_per_ee
        estimator_settings.force_threshold_down = 0.2 * robot_weight_per_ee
        self.estimator = RobotStateEstimator()
        self.estimator.initialize(estimator_settings)
        self.estimator.set_initial_state(self.q0, self.v0)

        #________ map to sensors ________#
        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')
        self.slider_positions = head.get_sensor('slider_positions')
        self.imu_gyroscope = head.get_sensor('imu_gyroscope')
        self.imu_accelerometer = head.get_sensor('imu_accelerometer')
        # self.observed_torques = head.get_sensor('joint_torques')

        #________ start gamepad thread ________#
        gamepad_thread = threading.Thread(target=gamepad_thread_fn)
        gamepad_thread.start()

        #________ initialze data logger ________#
        self.abs_log_path = None 
        if log_path is not None:
            self.abs_log_path = log_path 
            self.logger_file_name = str(self.abs_log_path+"/real_wobble_"
                        +deepcopy(time.strftime("%Y_%m_%d_%H_%M_%S")) + ".mds")
            self.logger = DataLogger(self.logger_file_name)
            # Input the data fields.
            self.id_time = self.logger.add_field("sim_time", 1)
            self.sim_q = self.logger.add_field("sim_q", self.pin_robot.nq)
            self.sim_v = self.logger.add_field("sim_v", self.pin_robot.nv)
            self.est_q = self.logger.add_field("est_q", self.pin_robot.nq)
            self.est_v = self.logger.add_field("est_v", self.pin_robot.nv)
            self.sim_imu_linacc = self.logger.add_field("sim_imu_linacc", 3)
            self.sim_imu_angvel = self.logger.add_field("sim_imu_angvel", 3)
            self.est_imu_linacc = self.logger.add_field("est_imu_linacc", 3)
            self.est_imu_angvel = self.logger.add_field("est_imu_angvel", 3) 
            self.applied_u = self.logger.add_field("applied_u", self.pin_robot.nv - 6)
            self.observed_u = self.logger.add_field("observed_u", self.pin_robot.nv - 6)
            self.sim_forces = {}
            self.est_forces = {}
            self.sim_contacts = {}
            self.est_contacts = {}
            for ee in self.robot_config.end_effector_names:
                self.sim_forces[ee] = self.logger.add_field("sim_" + ee + "_force", 6)
                self.est_forces[ee] = self.logger.add_field("est_" + ee + "_force", 6)
                self.sim_contacts[ee] = self.logger.add_field("sim_" + ee + "_contact", 1)
                self.est_contacts[ee] = self.logger.add_field("est_" + ee + "_contact", 1)

            self.logger.init_file()

    def log_data(self): 
        self.logger.begin_timestep() 
        self.logger.log(self.id_time, .01*self.t + .001*self.d)
        self.logger.log(self.sim_q, self.q_sim)
        self.logger.log(self.sim_v, self.v_sim)
        self.logger.log(self.est_q, self.q_est)
        self.logger.log(self.est_v, self.v_est)
        self.logger.log(self.sim_imu_linacc, self.imu_linacc_sim)
        self.logger.log(self.sim_imu_angvel, self.imu_angvel_sim)
        self.logger.log(self.est_imu_linacc, self.imu_linacc_est)
        self.logger.log(self.est_imu_angvel, self.imu_angvel_est)
        self.logger.log(self.applied_u, self.u_applied)
        self.logger.log(self.observed_u, self.u_observed)
        for i, ee in enumerate(self.robot_config.end_effector_names):
            self.logger.log(self.sim_forces[ee],self.f_sim[i])
            self.logger.log(self.est_forces[ee],self.f_est[i])
            self.logger.log(self.sim_contacts[ee], self.c_sim[i])
            self.logger.log(self.est_contacts[ee], self.c_est[i])
        self.logger.end_timestep()

    def map_sliders(self, sliders):
        sliders_out = np.zeros(12)
        slider_A = sliders[0]
        slider_B = sliders[1]
        for i in range(4):
            sliders_out[3 * i + 0] = slider_A
            sliders_out[3 * i + 1] = slider_B
            sliders_out[3 * i + 2] = 2. * (1. - slider_B)
            if i >= 2:
                sliders_out[3 * i + 1] *= -1
                sliders_out[3 * i + 2] *= -1
        # Swap the hip direction.
        sliders_out[3] *= -1
        sliders_out[9] *= -1
        return sliders_out

    def warmup(self, thread):
        # self.zero_pos = self.map_sliders(self.slider_positions)
        thread.vicon.bias_position(self.vicon_name)

    def get_base(self, thread):
        base_pos, base_vel = thread.vicon.get_state(self.vicon_name)
        base_vel[3:] = self.imu_gyroscope
        return base_pos, base_vel
    
    def read_forces(self, thread):
        self.f_sim[:,:] = thread.force_plate.get_contact_force(self.robot)

    def read_contact_status(self, thread): 
        self.c_sim[:] = thread.force_plate.get_contact_status(self.robot)
        # self.contact_status_flags[:] = [True if ci > 0.7 else False for ci in self.c_sim]

    def get_desired_state(self, thread):
        """ reads gamepad input and maps it to desired state, 
        returns:
            p_com: desired center of mass position 
            v_com: desired center of mass velocity
            b_ori: desired base orientation 
            b_ang_vel: desired base angular velocity 
            f_pos: desired end effector positions 
            f_vel: desired end effector velocites 

        """
        rotation_matrix =  pin.rpy.rpyToMatrix(gamepad_values)
        quaternion = pin.Quaternion(rotation_matrix).coeffs()     
        return self.com_ref, np.zeros(3), quaternion, np.zeros(3), self.contact_positions, np.zeros(6*len(self.contact_names))

    def run(self, thread):
        #________ read vicon, encoders, imu and force plate from Simulation ________#
        self.q_sim[:7], self.v_sim[:6] = self.get_base(thread)
        self.q_sim[7:] = self.joint_positions.copy()
        self.v_sim[6:] = self.joint_velocities.copy()
        self.x_sim[:self.pin_robot.nq] = self.q_sim
        self.x_sim[self.pin_robot.nq:] = self.v_sim
        self.imu_linacc_sim[:] = self.imu_accelerometer.copy()
        self.imu_angvel_sim[:] = self.imu_gyroscope.copy()  
        # if self.sim_forces:
        #     self.read_forces(thread)
        #     self.read_contact_status(thread)
        #________ Compute updated estimate from EKF ________#
        self.estimator.run(self.imu_linacc_sim, self.imu_angvel_sim, 
                           self.q_sim[7:], self.v_sim[6:], self.tau)
        self.estimator.get_state(self.q_est, self.v_est)
        c_est = self.estimator.get_detected_contact()
        self.c_est[:] = np.array([1 if ci else 0 for ci in c_est])
        for i,n in enumerate(self.contact_names):
            self.f_est[i,:3] = self.estimator.get_force(n)
        self.x_est[:self.pin_robot.nq] = self.q_est
        self.x_est[self.pin_robot.nq:] = self.v_est

        #________ Read Game Pad & construct Desired State ________#

        commands = self.get_desired_state(thread)

        com_position = pin.centerOfMass(self.pin_robot.model, self.pin_robot.data, self.q_sim)
        quat = pin.Quaternion(self.q_sim[3:7])
        #________ Run Actual Controller ________#
        self.controller.run(
            self.q_sim, self.v_sim,  # state feedback either actual or estimated 
            np.array([1., 1., 1., 1.]),  # active contacts 
            com_position, # actual center of mass position 
            commands[0], # desired center of mass position 
            quat.toRotationMatrix().dot(self.v_sim[:3]), # center of mass velocity in world frame 
            commands[1], # desired center of mass velocity in world frame 
            self.q_sim[3:7], # base orientation 
            commands[2], # desired base orientation as a quaternion (4d vector)
            self.v_sim[3:6], # base angular velocity 
            commands[3], # desired base angular velocity 
            commands[4], commands[5] # desired end effector positions and velocities 
        )

        self.tau[:] = self.controller.get_joint_torques()
        # for now assume perfect control, keep in mind this is not the case on the real robot 
        self.u_applied[:] = self.tau[:]
        self.u_observed[:] = self.tau[:] 

        #________ Log Data ________# 
        if self.abs_log_path is not None:
            self.log_data()

        #________ Increment Counter ________# 
        # self.d += 0.1 
        # if (self.d - 1.)**2 <= 1.e-8:
        #     self.d = 0. 
        #     self.t += 1 

        thread.head.set_control('ctrl_joint_torques', self.tau)



if __name__ == "__main__":
    #________ Create Simulation Environment ________#
    bullet_env = BulletEnvWithGround()
    robot = Solo12Robot()
    bullet_env.add_robot(robot) 
    pin_robot = Solo12Config.buildRobotWrapper() 

    p.resetDebugVisualizerCamera(1.6, 50, -35, (0.0, 0.0, 0.0))
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

    #________ Initialze Thread ________#
    head = SimHead(robot, vicon_name='solo12')
    safety_controller = HoldPDController(head, 3., 0.05, False)
    thread_head = ThreadHead(
        0.001, # dt.
        safety_controller, # Safety controllers.
        head, # Heads to read / write from.
        [     # Utils.
            ('vicon', SimVicon(['solo12/solo12'])),
            ('force_plate', SimForcePlate([robot]))
        ], 
        bullet_env # Environment to step.
    )

    #________ Centroidal Controller ________#
    log_path = None 

    q0 = Solo12Config.q0.copy()
    q0[0] = 0. 
    v0 =  Solo12Config.v0.copy()

    controller = CentroidalControlWithGamePad(head, 'solo12/solo12', 
                                    q0, v0, log_path) 

    safety_controller.zero_pos = q0[7:]
    
    thread_head.head.reset_state(q0, v0)
    # thread_head.switch_controllers(controller)

    # if log_path is None:
    #     thread_head.start_streaming()
    #     thread_head.start_logging()

    thread_head.sim_run(500)
    print("Switching Controller")
    thread_head.switch_controllers(controller)
    thread_head.sim_run(100000)

    # if log_path is None:
    #     thread_head.stop_streaming()
    #     thread_head.stop_logging()
    # else:
    #     controller.logger.close_file()
    # # Plot timing information.
    # thread_head.plot_timing() 
