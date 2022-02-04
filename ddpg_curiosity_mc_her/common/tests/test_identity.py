import pytest
from ddpg_curiosity_mc_her.common.tests.envs.identity_env import DiscreteIdentityEnv, BoxIdentityEnv
from ddpg_curiosity_mc_her.run import get_learn_function
from ddpg_curiosity_mc_her.common.tests.util import simple_test

common_kwargs = dict(
    total_timesteps=30000,
    network='mlp',
    gamma=0.9,
    seed=0,
)
   
learn_kwargs = {
    'a2c' : {},
    'acktr': {},
    'deepq': {},
    'ppo2': dict(lr=1e-3, nsteps=64, ent_coef=0.0),
    'trpo_mpi': dict(timesteps_per_batch=100, cg_iters=10, gamma=0.9, lam=1.0, max_kl=0.01)
}


@pytest.mark.slow
@pytest.mark.parametrize("alg", learn_kwargs.keys())
def test_discrete_identity(alg):
    '''
    Test if the algorithm (with an mlp policy)
    can learn an identity transformation (i.e. return observation as an action)
    '''

    kwargs = learn_kwargs[alg]
    kwargs.update(common_kwargs)

    learn_fn = lambda e: get_learn_function(alg)(env=e, **kwargs)
    env_fn = lambda: DiscreteIdentityEnv(10, episode_len=100)
    simple_test(env_fn, learn_fn, 0.9)

@pytest.mark.slow
@pytest.mark.parametrize("alg", ['a2c', 'ppo2', 'trpo_mpi'])
def test_continuous_identity(alg):
    '''
    Test if the algorithm (with an mlp policy)
    can learn an identity transformation (i.e. return observation as an action)
    to a required precision
    '''

    kwargs = learn_kwargs[alg]
    kwargs.update(common_kwargs)
    learn_fn = lambda e: get_learn_function(alg)(env=e, **kwargs)

    env_fn = lambda: BoxIdentityEnv((1,), episode_len=100)
    simple_test(env_fn, learn_fn, -0.1)

if __name__ == '__main__':
    test_continuous_identity('a2c')    

