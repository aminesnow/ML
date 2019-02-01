from DDQLearning import DDQLearning
from gameplay import Gameplay


ddq_learner = DDQLearning()
agents = ddq_learner.auto_play()

gp = Gameplay()

# gp.run_game()

agt1 = agents[0]
agt1.replay_memory()
agt1.update_target_weights()
agt1.replay_memory()

gp.run_game_with_agent(agt1)