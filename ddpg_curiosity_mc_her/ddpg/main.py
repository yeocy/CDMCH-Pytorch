import argparse
import time
import os
from ddpg_curiosity_mc_her import logger
from ddpg_curiosity_mc_her.ddpg.config import DEFAULT_PARAMS, get_convert_arg_to_type_fn, \
    log_params, prepare_params, create_agents, cached_make_env

import torch
import numpy as np
import random
import wandb
from mpi4py import MPI

def get_env_params(env, args):
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())
    assert len(env.action_space.shape) == 1

    if args['use_her']:
        assert len(obs['observation'].shape) == 1
        assert len(obs['desired_goal'].shape) == 1
        dims = {
            'obs': obs['observation'].shape[0],
            'action': env.action_space.shape[0],
            'goal': obs['desired_goal'].shape[0],
        }
    else:
        assert len(obs.shape) == 1
        dims = {
            'obs': obs.shape[0],
            'action': env.action_space.shape[0],
        }

    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]

    dims['action_max'] = env.action_space.high[0]
    dims['max_timesteps'] = env._max_episode_steps

    return dims

def run(args):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()

    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     wandb.init(project="CDMCH", entity="yeocy")

    if rank != 0:
        logger.set_level(logger.DISABLED)

    env = cached_make_env(args['make_env'])
    # Seed everything to make things reproducible.
    rank_seed = args['seed'] + 1000000 * rank
    myseed = rank_seed + 1000 * rank
    env.seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)

    # GPU setting
    gpu = args['cuda']
    if gpu:
        torch.cuda.manual_seed(myseed)
        torch.cuda.manual_seed_all(myseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info('rank {}: seed={}, logdir={}'.format(rank, rank_seed, logger.get_dir()))

    ###################################################################################
    # {'o': 40, 'u': 4, 'g': 9, 'info_is_success': 1, 'action_max': 1.0, 'max_timesteps': 100}
    input_dims = get_env_params(env, args)

    # exit()
    agents = create_agents(input_dims=input_dims, env=env, params=args)

    if MPI.COMM_WORLD.Get_rank() % 2 == 0:
        train_policy, role =agents['exploit'], 'exploit'
    else:
        train_policy, role =agents['explore'], 'explore'

    if rank == 0:
        start_time = time.time()

    train_policy.learn(role, agents)

    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


if __name__ == '__main__':

    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     wandb.init(project="CDMCH", entity="yeocy")
    # exit()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for key, value in DEFAULT_PARAMS.items():
        key = '--' + key.replace('_', '-')
        parser.add_argument(key, type=get_convert_arg_to_type_fn(type(value)), default=value)

    args = parser.parse_args()
    dict_args = vars(args)

    logger.configure()
    dict_args = prepare_params(dict_args)
    log_params(dict_args)

    run(dict_args)
