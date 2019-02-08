from DDQLearning import DDQLearning
from DDQNAgent import DDQNAgent
from gameplay import Gameplay
import gc
import random

N_EPISODES = 12000
AGT_NAME = 'Samantha'

def train_agent(n_episodes, start_eps=1.0, agt=None, resume=False):
    if resume:
        ddql = DDQLearning(AGT_NAME)
        ddql.agent = agt
        ddql.agent.epsilon = start_eps
    else:
        ddql = DDQLearning(AGT_NAME)
    return ddql.auto_play(n_episodes)


def load_agent(weights_name, name, with_eps=False):
    agt = DDQNAgent(name, with_eps)
    agt.load_weights(weights_name)
    return agt


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


gc.collect()


gp = Gameplay()


TRAIN_NEW_AGENT = True


if TRAIN_NEW_AGENT:
    agent = train_agent(N_EPISODES)
else:
    agent = load_agent('agt_{}.h5'.format(AGT_NAME), AGT_NAME, True)
    agent = train_agent(N_EPISODES, agt=agent, resume=True)

agent.save_weights('agt.h5')

#display_agent_memory(agent, 50)

agent.with_eps = False
gp.run_game_with_agent(agent)
