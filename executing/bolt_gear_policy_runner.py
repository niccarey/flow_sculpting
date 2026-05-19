import gym
import two_arm_insertion_environment

from rl_games.common import env_configurations # check rl games env requirements
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

import zmq # don't actually use this here, this is just a pass-through
import json
import yaml

import traceback
import logging 

# apparently we do need the network builder and A2C model to initialize and run the player
# Later: set up a stripped-back player function that is not dependent on A2C

from rl_components.my_network import MyNetworkBuilder 
from rl_components.my_a2c_model import MyA2CContinuousLogStd

from rl_components.my_agent import MyA2CAgent
from rl_components.my_player import MyPpoPlayerContinuous


config_name = "./bolt_gear_insertion_config.yaml" # Configuration file for the runner network, and shared context info

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

    model_builder.register_network('my_network', MyNetworkBuilder)
    model_builder.register_model('my_actor_model', lambda network, **kwargs: MyA2CContinuousLogStd(network))
    environment_name = "SimpleTwoArmInsertion-v0" # could punt this to config also

    logging.info("Launching policy controller")

    # THIS registers the environment with gym (and sets up the messaging socket for gym/araas communication)
    two_arm_insertion_environment.register_envs(address=msg_address)                  
    
    # THIS registers the environment with rl_games
    env_configurations.register("rlgpu", {"vecenv_type": "RAY", "env_creator": lambda **kwargs: env})
    env = gym.make(environment_name) 

    runner = Runner()

    # register an agent and player as an A2CAgent class, PPOPlayer class, respectively
    runner.algo_factory.register_builder('my_agent', lambda **kwargs : MyA2CAgent(**kwargs)) # still need A2C even though just playing
    runner.player_factory.register_builder('my_agent', lambda **kwargs : MyPpoPlayerContinuous(**kwargs))

    runner.load(config)

    # orchestrator: use environment to send an initial request for state information?
    # No, reset is autocalled by gym when running. So no need to hang before starting the runner
    runner.run(runner_args)
    logging.info("runner completed")

    # when we're done, close out the other end of the zmq pipeline
    env.close_socket()
