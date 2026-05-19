import gym
import simple_insertion_environment

from rl_games.common import env_configurations # check rl games env requirements
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

import math
import zmq
import copy

import json
import yaml

import traceback
import logging 
import sys


sys.path.append('/home/aidanc/multi-robot-assembly/multi_arm_assembly/')
from rl_components.my_models import ModelA2CContinuousLogStd
from rl_components.my_network_builder import A2CBuilder 
from rl_components.my_a2c_continuous import A2CAgent
from rl_components.my_players import PpoPlayerContinuous

config_name = "./elbow_insertion_config.yaml" # Configuration file for the runner network, and shared context info

with open(config_name, 'r') as stream:
    config = yaml.safe_load(stream)

# unpack shared parameters

log_file = config['params']['task']['log_name'] # to share logging info across both ends. (this is a debugging hack and may not be thread safe!)
checkpoint_path = config['params']['task']['checkpoint_file']
msg_address = config['params']['task']['socket_addr']

# Most network args are set on load through the config file, but some require run-time definition which we pass through a kwargs input:
runner_args = {'checkpoint' : checkpoint_path, 'play' : True, 'train': False, 'tf': False, 'num_actors':1, 'sigma': None, 'track': False,'wandb_project_name': 'rl_games' ,'wandb_entity': None, 'file': './bolt_gear_insertion_config.yaml', 'prefix': ''}

logging.basicConfig(filename=log_file,
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


if(__name__=="__main__"):

    model_builder.register_network('my_network', A2CBuilder)
    model_builder.register_model('my_actor_model', lambda network, **kwargs: ModelA2CContinuousLogStd(network))
    environment_name = "SimpleInsertion-v0" # could punt this to config also

    logging.info("Launching policy controller")

    # THIS registers the environment with gym (and sets up the messaging socket for gym/araas communication)
    simple_insertion_environment.register_envs(address=msg_address)                  
    
    # THIS registers the environment with rl_games
    env_configurations.register("rlgpu", {"vecenv_type": "RAY", "env_creator": lambda **kwargs: env})
    env = gym.make(environment_name) 

    runner = Runner()

    # register an agent and player as an A2CAgent class, PPOPlayer class, respectively
    runner.algo_factory.register_builder('my_agent', lambda **kwargs : A2CAgent(**kwargs)) # still need A2C even though just playing?
    # but it never gets initialized, so that's weird.

    # something is wrong in the ppo model params :/ Where do they pull from?
    runner.player_factory.register_builder('my_agent', lambda **kwargs : PpoPlayerContinuous(**kwargs))
    runner.load(config)

    # orchestrator: use environment to send an initial request for state information?
    # No, reset is autocalled by gym when running. So no need to hang before starting the runner

    # Batch size argument in player config - grr
    runner.run(runner_args) 
    logging.info("runner completed")

    # when we're done, close out the other end of the zmq pipeline
    # Why is this getting closed ahead of time?
    #env.close_socket()
