import gc

from utils.gameplay import Gameplay
from utils.learning_utils import LearningUtils

gc.collect()


N_EPISODES = 10000
TRAIN_NEW_AGENT = False

agt_name = LearningUtils.V2_NAME

if TRAIN_NEW_AGENT:
    agent = LearningUtils.train_agent(N_EPISODES, version=2)
else:
    agent = LearningUtils.load_agent('agt_{}_test.h5'.format(agt_name), version=2, with_eps=True)
    agent = LearningUtils.train_agent(N_EPISODES, start_eps=0.9, agt=agent, resume=True)

agent.save_weights('agt_{}_test.h5'.format(agt_name))
agent.with_eps = False

#display_agent_memory(agent, 50)

Gameplay.run_game_with_agent(agent)
