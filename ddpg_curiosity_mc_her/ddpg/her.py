import numpy as np
from mpi4py import MPI

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None, sub_goal_divisions = None):
        self.replay_strategy = replay_strategy#future
        self.replay_k = replay_k#4
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.sub_goal_divisions = sub_goal_divisions

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions # update normalizer : 100(50), self.batch_size : 100

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        if self.sub_goal_divisions is None:
            self.sub_goal_divisions_to_use = [range(transitions['g'].shape[1])]
        else:
            self.sub_goal_divisions_to_use = self.sub_goal_divisions
            assert sum([len(elem) for elem in self.sub_goal_divisions_to_use]) + 3 == episode_batch['g'].shape[2]

        for self.sub_goal_division in self.sub_goal_divisions_to_use:
            # Select future time indexes proportional with probability future_p. These
            # will be used for HER replay by substituting in future goals.

            # Choose which transitions from the batch we will alter
            her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

            # For each transition, choose a time offset between the max episode length and the time of the sample
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)

            # Get the future times to sample goals from (transitions not chosen for altering have original times)
            future_t = (t_samples + 1 + future_offset)[her_indexes]

            # Replace goal with achieved goal but only for the previously-selected
            # HER transitions (as defined by her_indexes). For the other transitions,
            # keep the original goal.

            future_achieved_goals = episode_batch['ag'][episode_idxs[her_indexes], future_t]

            transition_goals = transitions['g'][her_indexes]
            transition_goals[:, self.sub_goal_division] = future_achieved_goals[:, self.sub_goal_division]
            transitions['g'][her_indexes] = transition_goals

        if self.reward_func is not None:
            # Reconstruct info dictionary for reward computation.
            info = {}
            for key, value in transitions.items():
                if key.startswith('info_'):
                    info[key.replace('info_', '')] = value
            # Re-compute reward since we may have substituted the goal.
            reward_params = {k: transitions[k] for k in ['ag_next', 'g']}
            reward_params['info'] = info
            transitions['r'] = self.reward_func(reward_params['ag_next'], reward_params['g'], reward_params['info'])

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        assert (transitions['actions'].shape[0] == batch_size_in_transitions)
        return transitions