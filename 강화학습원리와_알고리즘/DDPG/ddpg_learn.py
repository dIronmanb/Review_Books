# DDPG learn (tf2 subclassing version: using chain to train Actor)
# coded by St.Watermelon

# import package
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
import tensorflow as tf

from replay_buffer import ReplayBuffer


# Actor
class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        
        self.action_bound = action_bound
        self.h1 = Dense(64, activation = 'relu')
        self.h1 = Dense(32, activation = 'relu')
        self.h1 = Dense(16, activation = 'relu')
        self.action = Dense(action_dim, activation = 'tanh')
        
        
    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        a = self.action(x)
        
        # action range -> [-action_bound, action_bound]
        a = Lambda(lambda x : x * self.action_bound)(a)
        
        return a
    
    
class Critic(Model):
    
    def __init__(self):
        super(Critic, self).__init__()
        
        self.x1 = Dense(32, activation = 'relu')
        self.a1 = Dense(32, activation = 'relu')
        self.h2 = Dense(32, activation = 'relu')
        self.h3 = Dense(16, activation = 'relu')
        self.q = Dense(1, activation = 'linear')
        
    def call(self, state_action):
        state = state_action[0]
        action = state_action[1]
        x = self.x1(state)
        a = self.a1(action)
        h = concatenate([x,a], axis = 1)
        x = self.h2(h)
        x = self.h3(x)
        q = self.q(x)
        return q
    

# DDPG Agent
class DDPG_Agent(object):
    
    def __init__(self, env):
        
        #hyper_parameter
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001 
        self.TAU = 0.001
        
        # environment
        self.env = env
        
        # state dimension
        self.state_dim = env.observation_space.shape[0]
        
        # action dimension
        self.action_dim = env.action_space.shape[0]
        
        # maximum size of action
        self.action_bound = env.action_space.high[0]
        
        # actor, target actor network, target critic network
        self.actor = Actor(self.action_dim, self.action_bound)
        self.target_actor = Actor(self.action_dim, self.action_bound)
        
        self.critic = Critic()
        self.target_critic = Critic()
        
        self.actor.build(input_shape = (None, self.state_dim))
        self.target_actor.build(input_shape = (None, self.state_dim))
        
        state_in = Input((self.state_dim))
        action_in = Input((self.action_dim))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])

        self.actor.summary()
        self.critic.summary()
        
        # Optimizer
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)
        
        # initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)
        
        # total reward
        self.save_epi_reward = []
        
        
    
    def update_target_network(self, TAU):
        theta = self.actor.get_weights()
        target_theta = self.target_actor.get_weights()
        for i in range(len(theta)):
            target_theta[i] = TAU * theta[i] + (1 - TAU) * target_theta[i]
        self.target_actor.set_weights(target_theta)
        
        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_theta[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_critic.set_weights(target_phi)
        
    
    def critic_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            q = self.critic([states, actions], training = True)
            loss = tf.reduce_mean(tf.square(q - td_targets))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        
    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states, training = True)
            critic_q = self.critic([states, actions])
            loss = -tf.reduce_mean(critic_q)
            
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients( zip(grads, self.actor.trainable_variables))
        
    def ou_noise(self, x, rho = 0.15, mu = 0, dt = 1e-1, sigma = 0.2 , dim = 1):
        return x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size = dim)
    
    
    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k
    
    
    def load_weights(self, path):
        self.actor.load_weights(path + '...')
        self.critic.load_weights(path + '...')
        
    # train agent
    def train(self, max_episode_num):
        
        # initialize target network
        self.update_target_network(1.0)
        
 
        for ep in range(int(max_episode_num)):
            
            # initialize OU noise
            pre_noise = np.zeros(self.action_dim)
            
            # initialize episodes
            time, episode_reward, done = 0, 0, False
            
            # initialize environment
            state = self.env.reset()
            
            while not done:
                
                self.env.render()
                
                action = self.actor(tf.convert_to_tensor([state], dtype = tf.float32))
                action = action.numpy()[0]
                
                noise = self.ou_noise(pre_noise, dim = self.action_dim)
                
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
                
                next_state, reward, done, _ = self.env.step(action)
                
                train_reward = (reward + 8) / 8
                
                self.buffer.add_buffer(state, action, train_reward, next_state, done)
                
                if self.buffer.buffer_count() > 1000:
                    
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)
                    
                    target_qs = self.target_critic([tf.convert_to_tensor(next_states, dtype =tf.float32),\
                                                    self.target_actor(tf.convert_to_tensor(\
                                                        tf.convert_to_tensor(next_states, dtype = tf.float32)))])
                    
                    y_i = self.td_target(rewards, target_qs.numpy(), dones)
                    
                    self.critic_learn(tf.convert_to_tensor(states, dtype = tf.float32) ,
                                      tf.convert_to_tensor(actions, dtype = tf.float32),
                                      tf.convert_to_tensor(y_i, dtype = tf.float32))
                    
                    self.actor_learn(tf.convert_to_tensor(states, dtype = tf.float32))
                    
                    self.update_target_network(self.TAU)
                    
                    
                    pre_noise = noise
                    state = next_state
                    episode_reward += reward
                    time += 1
                
                print('Episode: ', ep + 1, 'Time: ', time, 'Reward: ', episode_reward)
                self.save_epi_reward.append(episode_reward)
                
                self.actor.save_weights('./save_weights/...')
                self.critic.save_weights('./save_weights/')
                
                
            np.save.savetxt('./save_weights/...', self.save_epi_reward)
            print(self.save_epi_reward)
            
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
            
    