from DDQLearning import DDQLearning
from DDQNAgent import DDQNAgent
from gameplay import Gameplay
import gc
import random

gc.collect()

N_EPISODES = 40000

gp = Gameplay()

ddq_learner = DDQLearning()
agent = ddq_learner.auto_play(N_EPISODES)

agent.replay_memory()


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


# agent = DDQNAgent('Smith', False)
# agent.load_weights('agt.h5')
# gp.run_game_with_agent(agent)


# gp.run_game()
