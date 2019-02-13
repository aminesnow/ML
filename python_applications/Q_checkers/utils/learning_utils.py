import random

from utils.gameplay import Gameplay


class LearningUtils(object):

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

    @staticmethod
    def is_done_not_draw(reward, done):
        return done and (reward >= 1 or reward <= -1)
