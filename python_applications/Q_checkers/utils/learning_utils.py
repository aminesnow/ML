import random

from utils.gameplay import Gameplay
from v1.DDQAgent import DDQAgent as DDQAgentv1
from v2.DDQAgent import DDQAgent as DDQAgentv2


class LearningUtils(object):

    V1_NAME = 'Smith'
    V2_NAME = 'Samantha'

    @staticmethod
    def train_agent(n_episodes, start_eps=1.0, agt=None, version=1, resume=False):
        if resume:
            agt.epsilon = start_eps
        else:
            if version == 1:
                agt = DDQAgentv1(LearningUtils.V1_NAME, True)
            else:
                agt = DDQAgentv2(LearningUtils.V2_NAME, True)
        return agt.auto_play(n_episodes)

    @staticmethod
    def load_agent(weights_name, version=1, with_eps=False):
        if version == 1:
            agt = DDQAgentv1(LearningUtils.V1_NAME, with_eps)
        else:
            agt = DDQAgentv2(LearningUtils.V2_NAME, with_eps)
        agt.load_weights(weights_name)
        return agt

    @staticmethod
    def display_agent_memory(agt, size=None):
        if(size is None):
            size = len(agt.memory)
        minibatch = random.sample(agt.memory, size)
        for board_state, board_state_action, reward, next_board_state, next_possible_board_states, done in minibatch:
            if not done:
                print('board_state')
                Gameplay.show_board(board_state)
                print('board_state_action')
                Gameplay.show_board(board_state_action)
                print('next_board_state')
                Gameplay.show_board(next_board_state)
                print('next_possible_board_states')
                for poss in next_possible_board_states:
                    Gameplay.show_board(poss)
                print('reward {}'.format(reward))
                print('done {}'.format(done))
            else:
                print('board_state')
                Gameplay.show_board(board_state)
                print('board_state_action')
                Gameplay.show_board(board_state_action)
                print('reward {}'.format(reward))
                print('done {}'.format(done))