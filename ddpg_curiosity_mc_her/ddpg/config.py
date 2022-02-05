import gym
try:
    import gym_fetch_stack
except ImportError:
    pass

import os
import ast
from collections import OrderedDict
from ddpg_curiosity_mc_her import logger
from ddpg_curiosity_mc_her.ddpg.ddpg import DDPG
from mpi4py import MPI


DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
    'boxpush-v0': {
        'n_cycles': 10,
    },
}

DEFAULT_PARAMS = {
    'env_id': 'FetchReach-v1', # Try HalfCheetah-v2 for plain DDPG, FetchReach-v1 for HER
    'do_evaluation': True,
    'render_eval': False,
    'render_training': False,
    'seed': 42,
    'train_policy_fn': 'epsilon_greedy_noisy_explore',
    'eval_policy_fn': 'greedy_exploit',
    'agent_roles': 'exploit, explore',  # choices are 'explore, explore', 'exploit', and 'explore'
    'memory_type': 'replay_buffer',  # choices are 'replay_buffer' or 'ring_buffer'. 'ring_buffer' can't be used with HER.
    'heatmaps': False,  # generate heatmaps if using a gym-boxpush or FetchStack environment
    'boxpush_heatmaps': False,  # old argument, doesnt do anything, remaining to not break old scripts

    # networks
    'exploit_Q_lr': 0.001,  # critic learning rate
    'exploit_pi_lr': 0.001,  # actor learning rate
    'explore_Q_lr': 0.001,  # critic learning rate
    'explore_pi_lr': 0.001,  # actor learning rate
    'dynamics_lr': 0.007, # dynamics module learning rate
    'exploit_polyak_tau': 0.001,  # polyak averaging coefficient (target_net = (1 - tau) * target_net + tau * main_net)
    'explore_polyak_tau': 0.05,  # polyak averaging coefficient (target_net = (1 - tau) * target_net + tau * main_net)
    'exploit_gamma': 'auto',  # 'auto' or floating point number. If auto, gamma is 1 - 1/episode_time_horizon
    'explore_gamma': 'auto',  # 'auto' or floating point number. If auto, gamma is 1 - 1/episode_time_horizon
    'episode_time_horizon': 'auto',  # 'auto' or int. If 'auto' T is inferred from env._max_episode_steps

    # training
    'buffer_size': int(1E6),  # for experience replay
    'n_epochs': 25,
    'n_cycles': 50,  # per epoch
    'n_batches': 40,  # training batches per cycle
    'batch_size': 1024,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'rollout_batches_per_cycle': 8,
    'rollout_batch_size': 1,  # number of per mpi thread
    'n_test_rollouts': 50,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'noise_eps' : 0.2,
    'random_eps' : 0.3,

    # HER
    'use_her': True,
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    'sub_goal_divisions': 'none',

    # Save and Restore
    'save_at_score': .98,  # success rate for HER, mean reward per episode for DDPG
    'stop_at_score': 'none',  # success rate for HER, mean reward per episode for DDPG
    'save_checkpoints_at': 'none',
    'restore_from_ckpt': 'none',
    'do_demo_only': False,
    'demo_video_recording_name': 'none',

    # GPU Usage Overrides
    'cuda' : False,
    'num_gpu' : 'none',
}

CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):

    env_id = kwargs['env_id']

    def make_env():
        return gym.make(env_id)
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    kwargs['T'] = kwargs['episode_time_horizon']
    del kwargs['episode_time_horizon']
    if kwargs['T'] == 'auto':
        assert hasattr(tmp_env, '_max_episode_steps')
        kwargs['T'] = tmp_env._max_episode_steps
    else:
        kwargs['T'] = int(kwargs['T'])
    tmp_env.reset()

    if kwargs['use_her'] is False:
        # If HER is disabled, disable other HER related params.
        kwargs['replay_strategy'] = 'none'
        kwargs['replay_k'] = 0

    if 'BoxPush' not in kwargs['env_id'] and 'FetchStack' not in kwargs['env_id']:
        kwargs['heatmaps'] = False

    for gamma_key in ['exploit_gamma', 'explore_gamma']:
        kwargs[gamma_key] = 1. - 1. / kwargs['T'] if kwargs[gamma_key] == 'auto' else float(kwargs[gamma_key])

    # if kwargs['map_dynamics_loss'] and 'BoxPush' in kwargs['env_id'] and 'explore' in kwargs['agent_roles']:
    #     kwargs['dynamics_loss_mapper'] = DynamicsLossMapper(
    #             working_dir=os.path.join(logger.get_dir(), 'dynamics_loss'),
    #             sample_env=cached_make_env(kwargs['make_env'])
    #         )
    # else:
    #     kwargs['dynamics_loss_mapper'] = None

    # for network in ['exploit', 'explore']:
    #     # Parse noise_type
    #     action_noise = None
    #     param_noise = None
    #     nb_actions = tmp_env.action_space.shape[-1]
    #     for current_noise_type in kwargs[network+'_noise_type'].split(','):
    #         current_noise_type = current_noise_type.strip()
    #         if current_noise_type == 'none':
    #             pass
    #         elif 'adaptive-param' in current_noise_type:
    #             _, stddev = current_noise_type.split('_')
    #             param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
    #         elif 'normal' in current_noise_type:
    #             _, stddev = current_noise_type.split('_')
    #             action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    #         elif 'ou' in current_noise_type:
    #             _, stddev = current_noise_type.split('_')
    #             action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
    #                                                         sigma=float(stddev) * np.ones(nb_actions))
    #         else:
    #             raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
    #     kwargs[network+'_action_noise'] = action_noise
    #     kwargs[network+'_param_noise'] = param_noise
    #     del(kwargs[network+'_noise_type'])

    #TODO
    kwargs['train_rollout_params'] = {
        'compute_Q': False,
        'render': kwargs['render_training']
    }

    kwargs['eval_rollout_params'] = {
        'compute_Q': True,
        'render': kwargs['render_eval']
    }

    # if kwargs['mix_extrinsic_intrinsic_objectives_for_explore'] == 'none':
    #     kwargs['mix_extrinsic_intrinsic_objectives_for_explore'] = None
    # else:
    #     weights_string = kwargs['mix_extrinsic_intrinsic_objectives_for_explore']
    #     kwargs['mix_extrinsic_intrinsic_objectives_for_explore'] = [float(w) for w in weights_string.split(',')]
    #     assert len(kwargs['mix_extrinsic_intrinsic_objectives_for_explore']) == 2

    if kwargs['restore_from_ckpt'] == 'none':
        kwargs['restore_from_ckpt'] = None

    if kwargs['stop_at_score'] == 'none':
        kwargs['stop_at_score'] = None
    else:
        kwargs['stop_at_score'] = float(kwargs['stop_at_score'])

    if kwargs['sub_goal_divisions'] == 'none':
        kwargs['sub_goal_divisions'] = None
    else:
        sub_goal_string = kwargs['sub_goal_divisions']
        sub_goal_divisions = ast.literal_eval(sub_goal_string)

        assert type(sub_goal_divisions) == list
        for list_elem in sub_goal_divisions:
            assert type(list_elem) == list
            for index in list_elem:
                assert type(index) == int

        kwargs['sub_goal_divisions'] = sub_goal_divisions

    # if kwargs['split_gpu_usage_among_device_nums'] == 'none':
    #     kwargs['split_gpu_usage_among_device_nums'] = None
    # else:
    #     gpu_string = kwargs['split_gpu_usage_among_device_nums']
    #     gpu_nums = ast.literal_eval(gpu_string)
    #     assert len(gpu_nums) >= 1
    #     for gpu_num in gpu_nums:
    #         assert type(gpu_num) == int
    #     kwargs['split_gpu_usage_among_device_nums'] = gpu_nums

    # original_COMM_WORLD_rank = MPI.COMM_WORLD.Get_rank()
    # kwargs['explore_comm'] = MPI.COMM_WORLD.Split(color=original_COMM_WORLD_rank % kwargs['num_model_groups'],
    #                                               key=original_COMM_WORLD_rank)

    if kwargs['save_checkpoints_at'] == 'none':
        kwargs['save_checkpoints_at'] = None
    else:
        save_checkpoints_list = ast.literal_eval(kwargs['save_checkpoints_at'])
        assert type(save_checkpoints_list) == list
        for i in range(len(save_checkpoints_list)):
            save_checkpoints_list[i] = float(save_checkpoints_list[i])
        kwargs['save_checkpoints_at'] = save_checkpoints_list

    if kwargs["demo_video_recording_name"] == 'none':
        kwargs["demo_video_recording_name"] = None
    else:
        assert type(kwargs["demo_video_recording_name"]) == str

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))



def get_convert_arg_to_type_fn(arg_type):

    if arg_type == bool:
        def fn(value):
            if value in ['None', 'none']:
                return None
            if value in ['True', 'true', 't', '1']:
                return True
            elif value in ['False', 'false', 'f', '0']:
                return False
            else:
                raise ValueError("Argument must either be the string, \'True\' or \'False\'")
        return fn

    elif arg_type == int:
        def fn(value):
            if value in ['None', 'none']:
                return None
            return int(float(value))
        return fn
    elif arg_type == str:
        return lambda arg: arg
    else:
        def fn(value):
            if value in ['None', 'none']:
                return None
            return arg_type(value)
        return fn


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

def create_agents(input_dims, env, params):
    agent_roles = params['agent_roles'].replace(' ', '').split(',')
    agents = OrderedDict()

    if 'exploit' in agent_roles:
        role = 'exploit'
        agent = DDPG(role=role, input_dims=input_dims, env=env, params=params, external_critic_fn=None)
        agents[role] = agent
        # exploit_critic_fn = agent.critic_with_actor_fn
        logger.info('Using ' + role + ' agent.')
    else:
        exploit_critic_fn = None

    if 'explore' in agent_roles:
        role = 'explore'
        agent = DDPG(role=role, input_dims=input_dims, env=env, params=params)
        agents[role] = agent
        logger.info('Using ' + role + ' agent.')

    return agents


