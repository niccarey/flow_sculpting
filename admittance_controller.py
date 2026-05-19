import pyatk
import pyaraas
from pyatk import Transform, Vector
import time
import numpy as np
import math

class AdmittanceController():
    def __init__(self, mass_matrix, 
            kp_gains, 
            kd_gains, 
            force_limit = 100, 
            torque_limit = 1, 
            unit_scaling = 1000,
            max_vel = 20,
            max_rot_vel = 1):

        # gains
        self.M = mass_matrix
        self.kp = kp_gains
        self.kd = kd_gains
        self.force_limit = force_limit
        self.torque_limit = torque_limit
        self.unit_scale = unit_scaling
        self.max_v = max_vel 
        self.max_w = max_rot_vel


    def clip_input_wrench(self,world_wrench):

        world_force = Vector(world_wrench[0], world_wrench[1], world_wrench[2])
        world_torque = Vector(world_wrench[3], world_wrench[4], world_wrench[5])

        world_wrench_clip = np.array( world_wrench)

        if world_force.magnitude() > self.force_limit:
            world_wrench_clip[:3] = world_wrench[:3]*self.force_limit/(world_force.magnitude())

        if world_torque.magnitude() > self.torque_limit:
            world_wrench_clip[3:] = world_wrench[3:]*self.torque_limit/(world_torque.magnitude())

        return world_wrench_clip

    def get_vel_cmd(self, wrench, p_e, pd_e, time_step):
        force_output = wrench - np.multiply(self.kp, p_e) - np.multiply(self.kd, pd_e)
        accel_command = np.divide(force_output, self.M)

        # approximate instantaneous velocity
        # previous velocity is ignored to avoid "drift" - is drift actually windup, or due to internal approximations?
        vel_command = accel_command*time_step 

        # convert units:
        vel_command[:3] = vel_command[:3]*self.unit_scale
        
        # clamp the velocity and angular velocity command while maintaining "direction"
        linear_vel = Vector(vel_command[:3])
        rot_vel = Vector(vel_command[3:])

        if linear_vel.magnitude() > self.max_v:
            vel_command[:3] = vel_command[:3]*self.max_v/linear_vel.magnitude()

        if rot_vel.magnitude() > self.max_w:
            vel_command[3:] = vel_command[3:]*self.max_w/rot_vel.magnitude()

        return Vector(vel_command[:3]), Vector(vel_command[3:])