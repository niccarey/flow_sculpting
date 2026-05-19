import json
import asyncio
import gym
from gym import spaces,register
from gym.envs.registration import register

import numpy as np
import math
import traceback
import time
import zmq

import inspect
import logging
from collections import deque


def create_insertion_env(**kwargs):
    def _init():
        return SimpleInsertion(**kwargs)

    logging.info("Environment created")
    return _init

def register_envs(**kwargs):
    register(
        id='SimpleInsertion-v0',  # Unique ID for your environment
        entry_point=create_insertion_env(**kwargs),  # 'filename:classname'
        # max_episode_steps=100,  # Optional: Max steps per episode
    )

# Gym framework requirements:

# gym.Env.step(self, action: ActType)
# gym.Env.reset(self, *, seed: int | None = None, options: dict | None = None) → Tuple[ObsType, dict]
# gym.Env.render(self) → RenderFrame | List[RenderFrame] | None #(this should be none for running on araas)
# Env.action_space: Space[ActType]
# Env.observation_space: Space[ObsType]
# Env.reward_range = (-inf, inf)
# gym.Env.close(self) # can be omitted


# set up spaces is called any time before we invoke step
# We have a two-robot action space because this is a two-arm insertion: for a one-robot policy we can simplify this
# environment even further.
# Right now, one environment per insertion type seems easiest - avoids having to parse and adjust action/obs vector sizes 
# on the fly. If it's more efficient to re-use environments, can adjust the code.

# (note that state spaces are specific to each policy, and we will need to write new environments in any case to handle
# policies trained after October 24)

# UPDATES TO GOFLOW MAY HAVE AFFECTED ENVIRONMENT SETUP
# 

NUM_POSE_HISTORY = 10 # put this in config?

class SimpleInsertion(gym.Env):

    def __init__(self, address=None):        
        self.steps = 0 
        self.max_step = 30
        self.action_bound = 1
        self.obs_bound = 10
        self.pose_history = deque(maxlen=NUM_POSE_HISTORY)
        self.pose_history_size = 9*NUM_POSE_HISTORY # 9 observations per pose
        self.finished = False

        self.set_up_socket(address) 
        self.set_up_spaces()

    def set_up_socket(self, addr):
        if addr is None:
            logging.info("Error: environment must be registered with a valid communication socket address")

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(addr)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        self.socket.setsockopt(zmq.LINGER, 0)

        logging.info("Socket initialised")

    def close_socket(self):
        self.socket.close()
        self.ctx.term()

    def get_init_state(self):
        # this may be entirely subsumed by reset at run-time, but can use without resetting to hold until workcell is ready
        # send request to araas for state information
        init_msg = {}
        init_msg["update"] = True
        logging.info("Sending update request")
        if not self.finished:
            self.socket.send_json(init_msg) # ready for input

        poll_continue = True

        while poll_continue:
            timeout = 200 # set low to not hog the processor ? doesn't matter really
            socks = dict(self.poller.poll(timeout))
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                # hopefully received something
                logging.info("Receiving state")
                poll_continue = False
                break
        
        state_update_msg = self.socket.recv_json() 
        logging.info(state_update_msg)

        return state_update_msg

        
    def set_up_spaces(self):
        # could also hard code these in initialisation, but this looks cleaner
        # single insertion == 1 robots x 3 cartesian dimensions (could eventually go up to six per robot)
        self.action_dim = 3
        action_high = np.array([self.action_bound]*self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high)

        # Observation space dimensions are now incorrect. 
        # needs to include velocity and also a history buffer
        # velocity = 6 params 

        pose_dim = 9 # 9 observation dimensions per robot - pure pose info: 3x translation, 6x rotation
        vel_dim = 6
        history_dim = self.pose_history_size
        num_observations = pose_dim + vel_dim + history_dim

        obs_high = np.array([self.obs_bound] * num_observations)

        # no longer matches the network build unpacking method? BUT needs to be a dict to match the policy runner unpacking method?
        # ... why the hell is there a conflict here? oh ok. network used to have a different unpack policy
        #    if "policy" in input_shapes:
        #        input_shape = input_shapes["policy"][0]

        self.observation_space = spaces.Dict({"policy": spaces.Box(-obs_high, obs_high)})

    def get_RL_state(self, update_msg):
        all_obs =[] # should be a struct of np arrays, which we flatten /concatat the end.
        rstate_vecs = []
        obs_state = update_msg["robot_pose"]

        # left vs right extension could be a problem.
        
        # Need to populate the history buffer
        # and also send velocity information through the tunnel

        # ORDER: pose, velocity, historical
        rstate_vecs += [obs_state["trans"], obs_state["rot_c1"], obs_state["rot_c2"]]
        single_obs = np.concatenate(rstate_vecs)

        # seems dicey because we count it twice in the observations, but leave until we retrain at least.
        self.pose_history.extendleft([single_obs]) # should be a queue of np arrays, which we concatenate at the end.

        # First: pose

        all_obs.append(single_obs.flatten())

        # Compute and add velocity to observations
        linear_velocity, angular_velocity = self.compute_velocity(update_msg["robot_vel"])

        # then, lin, ang velocity
        all_obs.append(linear_velocity)
        all_obs.append(angular_velocity)

        if self.pose_history:
            #historical_poses = list(self.pose_history) # pose history should be a list already?
            padding_size = self.pose_history_size - len(np.array(self.pose_history).flatten()) #historical_poses.shape[1]
        else:
            padding_size = self.pose_history_size

        # array conversion gives 2D. can we flatten it?

        # so this is back to where it was - queue length is fucked up!
        if padding_size > 0:
            padding = np.zeros(padding_size)
            if padding_size < self.pose_history_size:
                historical_poses = np.concatenate([np.array(self.pose_history).flatten(), padding], axis = 0)
            else: 
                historical_poses = padding
        else:
            historical_poses = np.concatenate([np.array(self.pose_history).flatten()], axis=0)

        all_obs.append(historical_poses)
        obs = np.concatenate(all_obs, axis = 0)
        print("Observation")
        print(obs)

        return {"policy": obs} # mismatch between isaac and rl games expectations is screwing things up



    def step(self, action): 
        # this is fairly restricted - only reset takes optional arguments
        logging.info("received action: " + str([action]) ) 
        print("Action:")
        print(str([action]))

        self.steps+=1

        # compile into json message for sending
        action_msg = {}
        action_msg["robot_action"] = action[:3].tolist()

        # send action request
        self.socket.send_json(action_msg)
        logging.info("Action command sent to controller")

        # now listen for response:
        poll_continue = True

        while poll_continue:
            timeout = 200 # set low to not hog the processor ? doesn't matter really
            socks = dict(self.poller.poll(timeout))
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                # hopefully received something
                logging.info("Receiving observation")
                poll_continue = False
                break

        state_update_msg = self.socket.recv_json() 
        logging.info(state_update_msg)

        # Shape the returned state into an RL observation state

        try:
            state = self.get_RL_state(state_update_msg) 
        except Exception:
            print(traceback.format_exc())
            import sys
            sys.exit()
                
        done = self.steps > self.max_step

        if done:
            print("finished running")
            action_msg["done"] = True
            self.socket.send_json(action_msg)
            self.finished =True
            # Could close socket here ... why is reset getting called again?
            self.close_socket()

        # Where to add a default action + weighting? That has to happen on the robot side.

        # Expected return: obs, rewards, done, info
        # rewards need to be device cast even when zero, but can't do that here (?)
        # also 'done' state is not handled properly
        return state, 0, done, {}

    def compute_velocity(self, msg):
        linvel = np.array(msg['linear vel'])
        angvel = np.array(msg['angular vel'])
        return linvel, angvel
    
    def render(self, **kwargs):
        pass # hmm


    def reset(self, seed=None, options=None):
        logging.info("Env reset called")
        self.steps = 0
        self.pose_history.clear() 

        # state reset is not called explicitly by policy runner - how do we ensure starting states are initalized appropriately?
        state_message = self.get_init_state()
        
        # obs_shape needs to be dict wrapper, and observation needs to be dict.
        return self.get_RL_state(state_message)



