
import gym
import time
import random

# ==============================================================================
# ------------------------------------------------------------------------------
# returns the next action with the most reward as determined by MC
# don't call if env is done
# ------------------------------------------------------------------------------

def getMonteCarloAction(sim_size, depth):
    action_list = []
    orig_state = env.ale.cloneState()

    for i in range(env.action_space.n):
        _, reward, done, _ = env.step(i)
        if done:
            return reward

        mc_state = env.ale.cloneState()
        reward_list = []

        for _ in range(sim_size):
            reward_list.append( reward + __doMonteCarlo(depth) ) 
            env.ale.restoreState( mc_state )

        action_list.append( (sum(reward_list), i) ) # reward, index

        env.ale.restoreState(orig_state)

    assert(len(action_list) == env.action_space.n)

    action_list.sort()
    action_list.reverse()

    option_list = filter(lambda x: x[0] == action_list[0][0], action_list)

    return random.choice(option_list)[1]

# ==============================================================================
# ------------------------------------------------------------------------------
# returns the reward for a single MC simulation going depth steps ahead
# ------------------------------------------------------------------------------

def __doMonteCarlo(depth):
    reward_list = []

    for n in range(depth):
        action = env.action_space.sample()
        _, reward, done, _  = env.step(action)
        reward_list.append(reward)
        if done:
            break

    reward = sum(reward_list)
    return reward

# ==============================================================================

SIM_SIZE = 5
DEPTH    = 10
#ENV_NAME = 'CartPole-v0'
ENV_NAME = 'Pong-v0'

env = gym.make(ENV_NAME)

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = getMonteCarloAction(SIM_SIZE, DEPTH)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

