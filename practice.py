import gym
import gym_fetch_stack
import logging


env = gym.make("FetchStack3SparseStage1-v1")
env.reset()

# FetchStack2SparseStage1-v1
#
# FetchStack3SparseStage1-v1
#
# FetchStack4SparseStage1-v1
#
# FetchStack2SparseStage2-v1
#
# FetchStack3SparseStage2-v1
#
# FetchStack4SparseStage2-v1
#
# FetchStack2SparseStage3-v1
#
# FetchStack3SparseStage3-v1
#
# FetchStack4SparseStage3-v1
while True:
    env.render()