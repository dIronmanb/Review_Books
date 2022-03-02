import gym
from ddpg_learn import DDPG_Agent

def main():
    max_episode_num = 200 
    env = gym.make("...") 
    agent = DDPG_Agent(env)
    
    agent.train(max_episode_num)
    
    agent.plot_result()
    
if __name__ == "__main__":
    main()
