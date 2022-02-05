from functools import reduce

import numpy as np
from mpi4py import MPI
import torch
from datetime import datetime

from ddpg_curiosity_mc_her import logger
from ddpg_curiosity_mc_her.ddpg.her import her_sampler
from ddpg_curiosity_mc_her.ddpg.models import actor, critic, ForwardDynamics
from ddpg_curiosity_mc_her.ddpg.normalizer import normalizer
from ddpg_curiosity_mc_her.ddpg.mpi_utils import sync_networks, sync_grads
from ddpg_curiosity_mc_her.ddpg.replay_buffer import replay_buffer
import wandb

class DDPG(object):
    def __init__(self, role, input_dims, env, params, external_critic_fn=None):
        # Parameters.
        self.env_params = input_dims
        self.env = env
        self.cuda = params['cuda']
        self.n_epochs = params['n_epochs']
        self.n_cycles = params['n_cycles']
        self.noise_eps = params['noise_eps']
        self.random_eps = params['random_eps']
        self.n_batches = params['n_batches']
        self.rollout_batches_per_cycle = params['rollout_batches_per_cycle']
        self.gamma = params[role+'_gamma']
        self.polyak = params[role + '_polyak_tau']
        self.observation_range = (-5., 5.)
        self.goal_range = (-200, 200)
        self.actor_lr = params[role + '_pi_lr']
        self.critic_lr = params[role + '_Q_lr']
        self.batch_size = params['batch_size']
        self.role = role
        self.use_goals = True if params['use_her'] else False

        self.actor_network = actor(self.env_params, self.use_goals)
        self.critic_network = critic(self.env_params, self.use_goals)

        if self.role == 'explore':
            self.combined_actor_network = actor(self.env_params, self.use_goals)
        else:
            self.dynamic_network = ForwardDynamics(self.env_params, self.use_goals)

        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        if self.role =='explore':
            sync_networks(self.combined_actor_network)
        else:
            sync_networks(self.dynamic_network)

        # Target Network
        self.actor_target_network = actor(self.env_params, self.use_goals)
        self.critic_target_network = critic(self.env_params, self.use_goals)

        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # if use gpu
        if self.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
            if self.role == 'explore':
                self.combined_actor_network.cuda()
            else:
                self.dynamic_network.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)

        if self.role =='explore':
            self.combined_actor_optim = torch.optim.Adam(self.combined_actor_network.parameters(), lr=self.actor_lr)
        else:
            self.dynamic_optim = torch.optim.Adam(self.dynamic_network.parameters(), lr=self.actor_lr)

        # create the normalizer
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.observation_range)
        if self.use_goals:
            self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.goal_range)
        else:
            self.g_norm0 = None
            self.g_norm0 = None

        def reward_fun(ag, g, info):  # vectorized
            batch_size = np.shape(g)[0]
            rewards = env.compute_reward(achieved_goal=ag, desired_goal=g, info=info)
            return np.resize(rewards, new_shape=(batch_size, 1))
        # her sampler
        self.her_module = her_sampler(params['replay_strategy'],params['replay_k'], reward_func=reward_fun, sub_goal_divisions=params['sub_goal_divisions'])
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, params['buffer_size'], self.her_module.sample_her_transitions)

    def learn(self, role, agents):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(1, self.n_epochs+1):
            for _ in range(self.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions, mb_info = [], [], [], [], []
                for _ in range(self.rollout_batches_per_cycle):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions, ep_info = [], [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        # if MPI.COMM_WORLD.Get_rank() == 0:
                        #     self.env.render()
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, reward, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        info = [int(info['is_success']=='true')]
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        ep_info.append(info.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    mb_info.append(ep_info)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                mb_info = np.array(mb_info)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_info])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions, mb_info])

                for _ in range(self.n_batches):
                    # train the network
                    self._update_network(role, agents)
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # start to do the evaluation
            success_rate = self._eval_agent(agents)
            if MPI.COMM_WORLD.Get_rank() == 0:
                # wandb.log({"success rate": success_rate})
                logger.info('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                # print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                # torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                #             self.model_path + '/model.pt')


    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)

        if self.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])

        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], size=self.env_params['action'])

        # choose if use the random actions
        action += np.random.binomial(1, self.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, mb_info = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       'info': mb_info
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    # soft update
    def _soft_update_target_network(self, target, source):
        # 0.95
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.polyak) * param.data + self.polyak * target_param.data)

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.observation_range[0], self.observation_range[1])
        g = np.clip(g, -self.observation_range[0], self.observation_range[1])
        return o, g

    # update the network
    def _update_network(self, role, agent):
        # sample the episodes
        transitions = self.buffer.sample(self.batch_size)

        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        obs_norm_tensor = torch.tensor(obs_norm, dtype=torch.float32)
        obs_next_norm_tensor = torch.tensor(obs_next_norm, dtype=torch.float32)
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
            obs_next_norm_tensor = obs_next_norm_tensor.cuda()
            obs_norm_tensor = obs_norm_tensor.cuda()

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            if role == 'exploit':
                reward = torch.unsqueeze(torch.mean(torch.square(self.dynamic_network(obs_norm_tensor) - obs_next_norm_tensor), dim=1),1).detach()
                target_q_value = reward + self.gamma * q_next_value
                # print(r_tensor.shape) torch.Size([1024, 1])

            else:
                target_q_value = r_tensor + self.gamma * q_next_value
                # n_s = self.dynamics(obs_norm_tensor)

            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += (actions_real / self.env_params['action_max']).pow(2).mean()

        ## dynamic ##
        dynamic_loss = torch.mean(torch.square(agent['exploit'].dynamic_network(obs_norm_tensor) - obs_next_norm_tensor))
        agent['exploit'].dynamic_optim.zero_grad()
        dynamic_loss.backward()
        sync_grads(agent['exploit'].dynamic_network)
        agent['exploit'].dynamic_optim.step()

        combined_actions = agent['explore'].combined_actor_network(inputs_norm_tensor)
        exploit_Q = agent['exploit'].critic_network(inputs_norm_tensor, combined_actions)
        explore_Q = agent['explore'].critic_network(inputs_norm_tensor, combined_actions)
        combined_loss = torch.mean((exploit_Q + explore_Q)/2)

        agent['explore'].combined_actor_optim.zero_grad()
        combined_loss.backward()
        sync_grads(agent['explore'].combined_actor_network)
        agent['explore'].combined_actor_optim.step()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()


 # do the evaluation
    def _eval_agent(self, agent):
        total_success_rate = []
        for _ in range(10):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = agent['explore'].actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()