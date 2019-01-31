
from collections import deque
import numpy as np
import DNN

BOARD_SIZE = 8

class DDQNAgent:
    def __init__(self):
        self.memory = deque([], maxlen=4000)
        self.actions_dim = 4 #move piece from position (x,y) to (x',y')
        self.states_dim = BOARD_SIZE*BOARD_SIZE
        self.batch_size = TODO
        self.ddqn = DNN(self.states_dim,self.batch_size)
        self.ddqn_target = DNN(self.states_dim,self.batch_size)
        self.n_episodes = TODO
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_rate = 0.99
        self.loss = 0
        self.reward = 0
        self.reward_episode = 0
        self.target_update_threshold = 1000


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_memory(self):
        TODO

    def choose_action(self, state):
        possible_moves = TODO
        if np.random.rand() <= self.epsilon:
            action_index = self.np.random.randint(0, len(possible_moves))
        else:
            action_index = self.ddqn.best_action(state)
        return action_index

    def update_target_weights(self):
        self.ddqn_target.set_weights(self.ddqn.get_weights())
