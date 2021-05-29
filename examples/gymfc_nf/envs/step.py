import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gymfc.envs.fc_env import FlightControlEnv
import time
from .rewards import RewardEnv


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from gymfc_nf.policies import PpoBaselinesPolicy

class StepEnv(RewardEnv): 
    # def __init__(self, pulse_width = 1, max_rate = 100, state_fn = None,
                #  max_sim_time = 1 ):
    #Suvian 加入 max_angular
    def __init__(self, pulse_width = 1, max_rate = 100, max_angular=45, state_fn = None,
                 max_sim_time = 1 ):

        """Create a reinforcement learning environment that generates step input
        setpoints. Technically this is a multi-axis singlet input, the
        terminology in this package needs to be updated to reflect flight test
        maneuvers.

        This environment was created to teach an agent how to respond to
        worst-case inputs, that is, step inputs in which there is a request for
        immediate change in the target angular velocity.

        Start at zero deg/s to establish an initial condition and teach the
        agent to idle. Sample random input and hold for pulse_width, then
        return to zero deg/s to allow system to settle.

        Args:
            pulse_width: Number of seconds the step is held at the target 
                setpoint.
            max_rate: Max angular rate to sample from, or in the case of a 
                normal distribution, the mean. 
            state_fn: See BaseEnv
            max_sim_time: See BaseEnv
        """

        super().__init__(max_sim_time = max_sim_time, state_fn = state_fn)

        self.pulse_width = pulse_width
        self.max_rate = max_rate

        self.rising = True
        self.outputs = []
        self.angular_rate_sp = np.zeros(3)
        self.next_pulse_time = 0.512

        #Suvian
        self.attitude_sp = np.zeros(3)
        self.max_angular = max_angular
        self.rate_controller = None
        self.rate_error=np.zeros(3)
        self.last_rate_error=np.zeros(3)
        self.action_space = spaces.Box(-np.ones(3), np.ones(3), dtype=np.float32)
        self.sess=None

    def update_setpoint(self):
        # if self.sim_time > self.next_pulse_time:
        #     if (self.angular_rate_sp == np.zeros(3)).all():
        #         self.angular_rate_sp = self.generated_input
        #         self.next_pulse_time += self.pulse_width
        #     else:
        #         self.angular_rate_sp = np.zeros(3)
        #         self.next_pulse_time += self.pulse_width
        #     self.rising = False

        #Suvian
        if self.sim_time > self.next_pulse_time:
            if (self.attitude_sp == np.zeros(3)).all():
                self.attitude_sp = self.target_attitude
                self.next_pulse_time += self.pulse_width
            else:
                self.attitude_sp = np.zeros(3)
                self.next_pulse_time += self.pulse_width
            self.rising = False



    def reset(self):
        self.rising = True
        self.outputs = []
        self.angular_rate_sp = np.zeros(3)
        self.next_pulse_time = 0.512
        # Define the singlet input in the beginning so it can be overriden
        # externally if needed for testing.
        self.generated_input = self.sample_target()

        #Suvian
        self.attitude_sp = np.zeros(3)
        self.target_attitude=self.sample_target_attitude()

        return super().reset()

    def sample_target(self):
        """Sample a random angular velocity setpoint """
        return self.np_random.normal(0, self.max_rate, size=3)

    #Suvian
    def sample_target_attitude(self):
        return self.np_random.normal(0,self.max_angular,size=3)

    def state_rate_error_deltaerr(self):
        self.rate_error=self.angular_rate_sp - self.imu_angular_velocity_rpy
        error_delta = self.rate_error - self.last_rate_error 
        self.last_rate_error=self.rate_error
        return np.concatenate([self.rate_error, error_delta]) 

    def step(self,action):
        if(self.rate_controller==None):
            return
        self.angular_rate_sp = (action*500)
        print(action)
        ob = self.state_rate_error_deltaerr()
        with self.sess.as_default():
            ac = self.rate_controller.action(ob, self.sim_time, self.attitude_sp, self.attitude_rpy)
        # print(ac)
        return super().step(ac)
    
    def set_rate_controller(self,checkpoint_path):

        self.sess=tf.Session()
        with self.sess.as_default():
            saver = tf.train.import_meta_graph(checkpoint_path + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, checkpoint_path)
            self.rate_controller = PpoBaselinesPolicy(self.sess)
        
        print("----set_rate_controller successs")