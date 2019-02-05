from DDQLearning import DDQLearning
from DDQNAgent import DDQNAgent
from gameplay import Gameplay
import gc

N_EPISODES = 10000


def train_agent(n_episodes, start_eps=1.0, agt=None, resume=False):
    if resume:
        ddql = DDQLearning()
        ddql.agent = agt
        ddql.agent.epsilon = start_eps
    else:
        ddql = DDQLearning()
    return ddql.auto_play(n_episodes)


def load_agent(weights_name, name, with_eps=False):
    agt = DDQNAgent(name, with_eps)
    agt.load_weights(weights_name)
    return agt


gc.collect()


gp = Gameplay()


TRAIN_NEW_AGENT = False


if TRAIN_NEW_AGENT:
    agent = train_agent(N_EPISODES)
else:
    agent = load_agent('agt.h5', 'Smith', True)
    agent = train_agent(N_EPISODES, 0.5, agent, True)

agent.save_weights('agt.h5')


agent.with_eps = False
gp.run_game_with_agent(agent)


# minibatch = random.sample(agent.memory, 3)
# for board_state, board_state_action, reward, next_board_state, next_possible_board_states, done in agent.memory:
#     if not done:
#         print('board_state')
#         Gameplay.show_board(board_state)
#         print('board_state_action')
#         Gameplay.show_board(board_state_action)
#         print('next_board_state')
#         Gameplay.show_board(next_board_state)
#         print('next_possible_board_states')
#         for poss in next_possible_board_states:
#             Gameplay.show_board(poss)
#         print('reward {}'.format(reward))
#         print('done {}'.format(done))
