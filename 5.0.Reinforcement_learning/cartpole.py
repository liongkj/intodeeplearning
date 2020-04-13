import tensorflow as tf

import numpy as np
import base64
import io
import time
import gym
import IPython
import functools
import matplotlib.pyplot as plt
from tqdm import tqdm
import mitdeeplearning as mdl

# library from open ai for init a virtual env "CardPole": is a environment setting
env = gym.make("CartPole-v0")
# seed like some constant
env.seed(1)

# observation_space = variable that agent can observe
# 1. Cart position
# 2. Cart velocity
# 3. Pole angle
# 4. Pole rotation rate

n_observations = env.observation_space  # 4

# that can control
n_actions = env.action_space.n  # 2


### Define the Cartpole agent ###

# Defines a feed-forward neural network
def create_cartpole_model():
    model = tf.keras.models.Sequential([
        # First Dense layer
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=n_actions, activation=None)
    ])
    return model


cartpole_model = create_cartpole_model()

### Define the agent's action function ###

# Function that takes observations as input, executes a forward pass through model,
#   and outputs a sampled action.
# Arguments:
#   model: the network that defines our agent
#   observation: observation which is fed as input to the model
# Returns:
#   action: choice of agent action


def choose_action(model, observation):
    # add batch dimension to the observation
    observation = np.expand_dims(observation, axis=0)

    logits = model.predict(observation)

    # pass the log probabilities through a softmax to compute true probabilities
    prob_weights = tf.nn.softmax(logits).numpy()

    # TODO

    action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0]

    return action


### Define Agent Memory Class ###
class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    # So that agents remember what he did and is it good
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        '''TODO: update the list of actions with new action'''
        # TODO: your update code here
        self.actions.append(new_action)
        '''TODO: update the list of rewards with new reward'''
        # TODO: your update code here
        self.rewards.append(new_reward)


memory = Memory()


### Define Reward function ###

# Helper function that normalizes an np.array x
def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)  # standard deviation
    return x.astype(np.float32)

# Compute normalized, discounted, cumulative rewards (i.e., return)
# Arguments:
#   rewards: reward at timesteps in episode
#   gamma: discounting factor
# Returns:
#   normalized discounted reward

# if rewards length is 5


def discount_rewards(rewards, gamma=0.95):
    # init array of 0 based on rewards
    # [0,0,0,0,0]
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    # to make future rewards after n+ x to be 0
    # [5,4,3,2,1]
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)


### Define Loss function ###

# Arguments:
#   logits: network's predictions for actions to take
#   actions: the actions the agent took in an episode
#   rewards: the rewards the agent received in an episode
# Returns:
#   loss
def compute_loss(logits, actions, rewards):

    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)

    loss = tf.reduce_mean(rewards*neg_logprob)
    return loss


### Define Training step (forward and backpropagation) ###

def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network
        logits = model(observations)

        # call loss function above
        loss = compute_loss(logits, actions, discounted_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


### Cartpole training! ###

# Learning rate and optimizer
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

# instantiate cartpole agent
cartpole_model = create_cartpole_model()

# to track our progress
smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
plotter = mdl.util.PeriodicPlotter(
    sec=2, xlabel='Iterations', ylabel='Rewards')

if hasattr(tqdm, '_instances'):
    tqdm._instances.clear()  # clear if it exists
for i_episode in range(500):

    plotter.plot(smoothed_reward.get())

    # Restart the environment
    observation = env.reset()
    memory.clear()

    while True:
        # using our observation, choose an action and take it in the environment
        action = choose_action(cartpole_model, observation)
        next_observation, reward, done, info = env.step(action)
        # add to memory
        memory.add_to_memory(observation, action, reward)

        # is the episode over? did you crash or do so well that you're done?
        if done:
            # determine total reward and keep a record of this
            total_reward = sum(memory.rewards)
            smoothed_reward.append(total_reward)

            # initiate training - remember we don't know anything about how the
            #   agent is doing until it has crashed!
            train_step(cartpole_model, optimizer,
                       observations=np.vstack(memory.observations),
                       actions=np.array(memory.actions),
                       discounted_rewards=discount_rewards(memory.rewards))

            # reset the memory
            memory.clear()
            break
        # update our observatons
        observation = next_observation

saved_cartpole = mdl.lab3.save_video_of_model(cartpole_model, "CartPole-v0")
mdl.lab3.play_video(saved_cartpole)
