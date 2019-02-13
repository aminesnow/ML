import gc
from utils.gameplay import Gameplay
from utils.agent_utils import AgentUtils

gc.collect()


N_EPISODES = 10000
TRAIN_NEW_AGENT = False

agt_name = AgentUtils.V2_NAME

if TRAIN_NEW_AGENT:
    agent = AgentUtils.train_agent(N_EPISODES, version=2)
else:
    agent = AgentUtils.load_agent('agt_{}_test.h5'.format(agt_name), version=2, with_eps=True)
    agent = AgentUtils.train_agent(N_EPISODES, start_eps=0.9, agt=agent, resume=True)

agent.save_weights('agt_{}_test.h5'.format(agt_name))
agent.with_eps = False

#display_agent_memory(agent, 50)

Gameplay.run_game_with_agent(agent)
