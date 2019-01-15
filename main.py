from __future__ import division, print_function, unicode_literals

import argparse
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from tensorboardX import SummaryWriter

import gym
import numpy as np
from gym import wrappers

import torch
from ddpg import DDPG
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition

def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--exp-name', default='naf-ikostrikov',
                        help='using NAF by Ikostrikov')
    parser.add_argument('--algo', default='NAF',
                        help='algorithm to use: DDPG | NAF')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='name of the environment to run')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--ou_noise', type=bool, default=True)
    parser.add_argument('--param_noise', type=bool, default=False)
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial noise scale (default: 0.3)')
    parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                        help='final noise scale (default: 0.3)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=4, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='number of episodes (default: 128)')
    parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    args = parser.parse_args()
    return args

def train(args, logger):

    env = NormalizedActions(gym.make(args.env_name))

    writer = SummaryWriter()

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.algo == "NAF":
        agent = NAF(args.gamma, args.tau, args.hidden_size,
                          env.observation_space.shape[0], env.action_space)
    else:
        agent = DDPG(args.gamma, args.tau, args.hidden_size,
                          env.observation_space.shape[0], env.action_space)

    memory = ReplayMemory(args.replay_size)

    ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,
        desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05) if args.param_noise else None

    rewards = []
    total_numsteps = 0
    updates = 0

    for i_episode in range(args.num_episodes):
        state = torch.Tensor([env.reset()])

        if args.ou_noise:
            ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                          i_episode) / args.exploration_end + args.final_noise_scale
            ounoise.reset()

        if args.param_noise and args.algo == "DDPG":
            agent.perturb_actor_parameters(param_noise)

        episode_reward = 0
        episode_step_counter = 0
        episode_rewards_train = []
        while True:
            episode_step_counter += 1
            action = agent.select_action(state, ounoise, param_noise)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            action = torch.Tensor(action)
            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward])

            memory.push(state, action, mask, next_state, reward)

            state = next_state
            print("next state: {}".format(episode_step_counter))

            if len(memory) > args.batch_size:
                for _ in range(args.updates_per_step):
                    transitions = memory.sample(args.batch_size)
                    batch = Transition(*zip(*transitions))

                    value_loss, policy_loss = agent.update_parameters(batch)

                    writer.add_scalar('loss/value', value_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)

                    updates += 1
            if done:
                print("episode step counter (final): {}".format(episode_step_counter))
                break
        episode_rewards_train.append(episode_reward)
        writer.add_scalar('reward/train', episode_reward, i_episode)

        # Update param_noise based on distance metric
        if args.param_noise:
            episode_transitions = memory.memory[memory.position-t:memory.position]
            states = torch.cat([transition[0] for transition in episode_transitions], 0)
            unperturbed_actions = agent.select_action(states, None, None)
            perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

            ddpg_dist = ddpg_distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
            param_noise.adapt(ddpg_dist)

        rewards.append(episode_reward)
        if i_episode % 10 == 0:
            prefix = ""
            logger.log_stat(prefix + "return_mean",
                            np.mean(episode_rewards_train[-10:]),
                            total_numsteps)

            state = torch.Tensor([env.reset()])
            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.numpy()[0])
                episode_reward += reward

                next_state = torch.Tensor([next_state])

                state = next_state
                if done:
                    break

            writer.add_scalar('reward/test', episode_reward, i_episode)

            rewards.append(episode_reward)
            print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, rewards[-1], np.mean(rewards[-10:])))
            prefix = "test"
            logger.log_stat(prefix + "return_mean",
                            np.mean(rewards[-10:]),
                            total_numsteps)
            pass
    env.close()

# additional code to make codebase sacred compatible
#ßfrom __future__ import division, print_function, unicode_literals
from sacred import Experiment
import numpy as np
import os
import collections
from os.path import dirname, abspath
import pymongo
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
from utils.logging import get_logger, Logger
from utils.dict2namedtuple import convert
import yaml
import os


SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

mongo_client = None


# Function to connect to a mongodb and add a Sacred MongoObserver
def setup_mongodb(db_url, db_name):
    client = None
    mongodb_fail = True

    # Try 5 times to connect to the mongodb
    for tries in range(5):
        # First try to connect to the central server. If that doesn't work then just save locally
        maxSevSelDelay = 10000  # Assume 10s maximum server selection delay
        try:
            # Check whether server is accessible
            logger.info("Trying to connect to mongoDB '{}'".format(db_url))
            client = pymongo.MongoClient(db_url, ssl=True, serverSelectionTimeoutMS=maxSevSelDelay)
            client.server_info()
            # If this hasn't raised an exception, we can add the observer
            ex.observers.append(MongoObserver.create(url=db_url, db_name=db_name, ssl=True)) # db_name=db_name,
            logger.info("Added MongoDB observer on {}.".format(db_url))
            mongodb_fail = False
            break
        except pymongo.errors.ServerSelectionTimeoutError:
            logger.warning("Couldn't connect to MongoDB on try {}".format(tries + 1))

    if mongodb_fail:
        logger.error("Couldn't connect to MongoDB after 5 tries!")
        # TODO: Maybe we want to end the script here sometimes?

    return client


@ex.main
def my_main(_run, _config, _log):
    global mongo_client

    import datetime
    unique_token = "{}__{}".format(_config["name"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # run the framework
    # run(_run, _config, _log, mongo_client, unique_token)
    arglist = parse_args()

    logger = Logger(_log)
    # configure tensorboard logger
    unique_token = "{}__{}".format(arglist.exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    use_tensorboard = False
    if use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    logger.setup_sacred(_run)
    train(arglist, logger)
    # arglist = convert(_config)
    #train(arglist)

    # force exit
    os._exit(0)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

if __name__ == '__main__':
    import os

    arglist = parse_args()

    from copy import deepcopy
    # params = deepcopy(sys.argv)

    # scenario_name = None
    # for _i, _v in enumerate(params):
    #     if _v.split("=")[0] == "--scenario":
    #         #scenario_name = _v.split("=")[1]
    #         scenario_name = params[_i + 1]
    #         del params[_i:_i+2]
    #         break
    #
    # name = None
    # for _i, _v in enumerate(params):
    #     if _v.split("=")[0] == "--name":
    #         #scenario_name = _v.split("=")[1]
    #         name = params[_i + 1]
    #         del params[_i:_i+2]
    #         break

    # now add all the config to sacred
    # ex.add_config({"scenario":scenario_name,
    #                "name":name})
    ex.add_config({"name":arglist.exp_name})

    # Check if we don't want to save to sacred mongodb
    no_mongodb = False

    # for _i, _v in enumerate(params):
    #     if "no-mongo" in _v:
    #     # if "--no-mongo" == _v:
    #         del params[_i]
    #         no_mongodb = True
    #         break

    config_dict={}
    config_dict["db_url"] = "mongodb://pymarlOwner:EMC7Jp98c8rE7FxxN7g82DT5spGsVr9A@gandalf.cs.ox.ac.uk:27017/pymarl"
    config_dict["db_name"] = "pymarl"

    # If there is no url set for the mongodb, we cannot use it
    if not no_mongodb and "db_url" not in config_dict:
        no_mongodb = True
        logger.error("No 'db_url' to use for Sacred MongoDB")

    if not no_mongodb:
        db_url = config_dict["db_url"]
        db_name = config_dict["db_name"]
        mongo_client = setup_mongodb(db_url, db_name)

    # Save to disk by default for sacred, even if we are using the mongodb
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline("")
