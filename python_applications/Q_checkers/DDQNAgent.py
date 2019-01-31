from collections import deque
import numpy as np
from .DNN import DNN
import random


BOARD_SIZE = 8
MEM_SIZE = 4000


class DDQNAgent(object):
    def __init__(self):
        self.memory = deque([], maxlen=MEM_SIZE)
        self.actions_dim = 4 #move piece from position (x,y) to (x',y')
        self.state_size = BOARD_SIZE*BOARD_SIZE
        self.ddqn = DNN(self.state_size,self.batch_size)
        self.ddqn_target = DNN(self.state_size,self.batch_size)
        self.episodes_count = 0
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_rate = 0.99
        self.loss = 0
        self.reward = 0
        self.reward_episode = 0
        self.target_update_threshold = 200

    def remember(self, board_state, action, reward, next_board_state, next_possible_moves, done):
        self.memory.append((board_state, action, reward, next_board_state, next_possible_moves, done))

    def replay_memory(self, batch_size):
        minibatch = random.sample(self.memory)
        for board_state, action, reward, next_board_state, next_possible_moves, done in minibatch:
            if done:
                q_value = reward
            else:
                targets = []
                for next_move in next_possible_moves:
                    targets.append(self.ddqn.predict_Q(next_board_state, next_move)[0])
                next_best_action = next_possible_moves[np.argmax(targets)]
                q_value_t = self.ddqn_target.predict_Q(next_board_state,next_best_action)[0]
                q_value = reward + self.gamma * q_value_t

            board_state_reshaped = board_state.reshape((1, self.state_size))
            self.ddqn.train(np.hstack((board_state_reshaped, action)), q_value)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_rate

    def choose_action(self, board_state, possible_moves):
        if np.random.rand() <= self.epsilon:
            action_index = np.random.randint(0, len(possible_moves))
        else:
            q_values = []
            for move in possible_moves:
                q_values.append(self.ddqn.predict_Q(board_state, move)[0])
            action_index = np.argmax(q_values)
        return action_index

    def update_target_weights(self):
        self.ddqn_target.model.set_weights(self.ddqn.model.get_weights())

