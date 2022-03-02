import gym
from ddpg_learn import DDPG_Agent
import tensorflow as tf


def main():
    env = gym.make('...')
    agent = DDPG_Agent(env)
    agent.load_weights('./save_weights/')
    
    time = 0
    state = env.reset()
    
    while True:
        env.render()
        
        action = agent.actor(tf.convert_to_tensor([state], dtype = tf.float32)).numpy()[0]
        
        state, reward, done, _ = env.step(action)
        
        time += 1
        
        print('Time: ', time, 'Reward: ', reward)
        
        if done:
            break
        
    env.close()
    
if __name__  == "__main__":
    main()