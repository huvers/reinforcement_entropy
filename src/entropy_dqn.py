import sys
import gym
from dqn import Agent

num_episodes = 20

#env_name = sys.argv[1] if len(sys.argv) > 1 else "MsPacman-v0"
env_name = sys.argv[1] if len(sys.argv) > 1 else "Pong-v0"
env = gym.make(env_name)
dummy_env = gym.make(env_name)

agent = Agent(state_size=env.observation_space.shape,
              number_of_actions=env.action_space.n,
              save_name=env_name)

print 'Unique Actions for Action Space: ', env.action_space.n
print 'State Space Size of Environment: ', env.observation_space

depth = 3  # distance into the future you want to sample
sim_size = 50  # num of times to simulate total
maxTreeSteps = env.action_space.n**depth
print 'Max Number of Possible Steps: ', maxTreeSteps


def getMonteCarlo(action, depth):
    # take initial action, then follow with random actions...
    observation_mc, reward_mc, done_mc, info_mc = env.step(action)
    temp_list = []
    for n in range(depth):
        action_mc = env.action_space.sample()  # get random action
        observation_mc, reward_mc, done_mc, info_mc = env.step(action_mc)  # apply random action to env
        temp_list.append(reward_mc)
        if done_mc:
            reward = sum(temp_list)
            return reward
    reward = sum(temp_list)
    return reward


for e in xrange(num_episodes):
    observation = env.reset()
    print 'New Episode...'
    done = False
    agent.new_episode()
    total_cost = 0.0
    total_reward = 0.0
    frame = 0
    while not done:
        frame += 1
        env.render()

        action_list = []

        for i in range(env.action_space.n):
            reward_list = []
            for j in range(sim_size):
                reward_list.append(getMonteCarlo(i, depth))
            action_list.append(sum(reward_list))
        #print 'Action List: ', action_list
        action_index = sorted(range(len(action_list)), key=lambda k: action_list[k], reverse=True)
        #print 'Sorted Index: ', action_index

        best_action = action_index[0]
        action, values = agent.act(observation)
        print 'Action: ', best_action
        observation, reward, done, info = env.step(best_action)
        print done
    #    print 'Reward: ', reward
        total_cost += agent.observe(reward)
        total_reward += reward

    print "total reward", total_reward
    print "mean cost", total_cost/frame
