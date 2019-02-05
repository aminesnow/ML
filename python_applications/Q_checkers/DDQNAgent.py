from collections import deque
import numpy as np
from DNN import DNN
import random
from gameplay import Gameplay


BOARD_SIZE = 8
MEM_SIZE = 4000
WEIGHTS_DIR = './weights/'


class DDQNAgent(object):
    def __init__(self, name, with_eps):
        self.name = name
        self.memory = deque([], maxlen=MEM_SIZE)
        #self.actions_dim = 4 #move piece from position (x,y) to (x',y')
        self.state_size = BOARD_SIZE*BOARD_SIZE
        self.replay_batch_size = 254
        self.ddqn = DNN(self.state_size)
        self.ddqn_target = DNN(self.state_size)
        self.minibatch_count = 0
        self.gamma = 0.999
        self.epsilon = 1.0
        self.epsilon_min = 0.03
        self.epsilon_decay_rate = 0.995
        self.target_update_threshold = 1016
        self.replay_count = 0
        self.with_eps = with_eps
        self.loss_mean = 0

    def remember(self, board_state, board_state_action, reward, next_board_state, next_possible_board_states, done):
        self.memory.append((board_state, board_state_action, reward, next_board_state, next_possible_board_states, done))

    def replay_memory(self):
        print('Replay memory!')
        samples = random.sample(self.memory, self.replay_batch_size)
        avg_loss = 0
        minibatch_X = []
        minibatch_y = []
        for board_state, board_state_action, reward, next_board_state, next_possible_board_states, done in samples:
            if(self.replay_count == self.target_update_threshold):
                self.update_target_weights()
                self.replay_count = 0

            if done or next_possible_board_states is None or not next_possible_board_states.size > self.state_size:
                q_value = reward
            else:
                targets = []
                for possible_board_state in next_possible_board_states:
                    targets.append(self.ddqn.predict_Q(next_board_state, possible_board_state)[0])

                targets = np.array(targets)
                next_best_board = next_possible_board_states[np.argmax(targets)]
                q_value_t = self.ddqn_target.predict_Q(next_board_state, next_best_board)[0]
                q_value = reward + self.gamma * q_value_t

            board_state_reshaped = board_state.reshape(self.state_size)
            board_state_action_reshaped = board_state_action.reshape(self.state_size)
            minibatch_X.append(np.hstack((board_state_reshaped, board_state_action_reshaped)))
            minibatch_y.append(np.clip(q_value, -2, 2))
            self.replay_count += 1

        self.minibatch_count += 1
        #print(minibatch_X)
        hist = self.ddqn.train(np.array(minibatch_X), np.array(minibatch_y))
        avg_loss += np.mean(hist.history['loss'])

        self.loss_mean = (self.loss_mean*self.minibatch_count + avg_loss)/(self.minibatch_count+1)

        print('{} minibatch average Q_value: {}'.format(self.name, np.mean(minibatch_y)))
        print('{} minibatch average loss: {}'.format(self.name, avg_loss))
        print('{} overall average loss: {}'.format(self.name, self.loss_mean))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_rate
        print('{} epsilon: {}'.format(self.name, self.epsilon))

    def choose_action(self, board_state, possible_board_states, invert=False):
        # self.with_eps = False
        q_values = []
        if invert:
            board_state = Gameplay.invert_board(board_state)
            possible_board_states = np.array(list(map(lambda x: Gameplay.invert_board(x), possible_board_states)))
        if self.with_eps and np.random.rand() <= self.epsilon:
            #print('Random action!')
            action_index = np.random.randint(0, len(possible_board_states))
            q_values.append(0)
        else:
            #print('Greedy action!')
            #Gameplay.show_board(board_state)
            #print('-----{}-----'.format(invert))
            for possible_bd_state in possible_board_states:
                #Gameplay.show_board(possible_bd_state)
                #print('board q: {}'.format(self.ddqn.predict_Q(board_state, possible_bd_state)[0]))
                q_values.append(self.ddqn.predict_Q(board_state, possible_bd_state)[0])

            action_index = np.argmax(q_values)
        return action_index, np.max(q_values)

    def get_moves_Q_values(self, board_state, possible_board_states, invert=False):
        q_values = []
        if invert:
            board_state = Gameplay.invert_board(board_state)
            possible_board_states = np.array(list(map(lambda x: Gameplay.invert_board(x), possible_board_states)))

        for possible_bd_state in possible_board_states:
            # Gameplay.show_board(possible_bd_state)
            q_values.append(self.ddqn.predict_Q(board_state, possible_bd_state)[0])

        return q_values

    def update_target_weights(self):
        print('Update target weights!')
        self.ddqn_target.model.set_weights(self.ddqn.model.get_weights())

    def save_weights(self, filename):
        self.ddqn.model.save_weights(WEIGHTS_DIR+filename, overwrite=True)
        self.update_target_weights()

    def load_weights(self, filename):
        self.ddqn.model.load_weights(WEIGHTS_DIR+filename)
