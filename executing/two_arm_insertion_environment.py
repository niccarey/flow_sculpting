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

def create_insertion_env(**kwargs):
    def _init():
        return SimpleTwoArmInsertion(**kwargs)

    logging.info("Environment created")
    return _init

def register_envs(**kwargs):
    register(
        id='SimpleTwoArmInsertion-v0',  # Unique ID for your environment
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


class SimpleTwoArmInsertion(gym.Env):

    def __init__(self, address=None):        
        self.steps = 0 
        self.max_step = 30
        self.action_bound = 1
        self.obs_bound = 10

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
        
        # two arm insertion == 2 robots x 3 cartesian dimensions (could eventually go up to six each)
        self.action_dim = 6
        action_high = np.array([self.action_bound]*self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high)

        obs_dim = 18 # 9 observation dimensions per robot - pure pose, 3x translation, 6x rotation

        obs_high = np.array([self.obs_bound] * obs_dim)
        self.observation_space = spaces.Dict({"policy": spaces.Box(-obs_high, obs_high)})

    def get_RL_state(self, update_msg):

        rstate_vecs = []
        r0_obs_state = update_msg["r0_pose"]
        r1_obs_state = update_msg["r1_pose"]

        # unpack into a single vector comprising (translation vector, first two cols of rotation matrix)
        for robot_state in [r0_obs_state, r1_obs_state]:
            rstate_vecs+=[robot_state["trans"], robot_state["rot_c1"], robot_state["rot_c2"]]

        obs = np.concatenate(rstate_vecs)
        return {"policy": obs}


    def step(self, action): 
        # this is fairly restricted - only reset takes optional arguments

        logging.info("received action: " + str([action]) ) 
        self.steps+=1
        # we expect the incoming action to be 3x action for robot 0, 3x action for robot 1
        # in other environments there may be additional rotational dofs, etc

        # compile into json message for sending
        action_msg = {}
        action_msg["r0_action"] = action[:3].tolist()
        action_msg["r1_action"] = action[3:].tolist()

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
            action_msg["done"] = True
            self.socket.send_json(action_msg)
        
        return state, 0, done, {}

    
    def render(self, **kwargs):
        pass # hmm


    def reset(self, seed=None, options=None):
        logging.info("Env reset called")
        self.steps = 0

        # state reset is not called explicitly by policy runner - how do we ensure starting states are initalized appropriately?
        state_message = self.get_init_state()
        
        return self.get_RL_state(state_message)



